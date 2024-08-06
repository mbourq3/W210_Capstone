import streamlit as st
import requests
#import streamlit.components.v1 as components
import math
import pandas as pd
import joblib
import time
import category_encoders
from category_encoders import TargetEncoder
import xgboost
import numpy as np
import os
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Estimate your solar power requirements")

st.title("SmartSolarSizer")
st.write("Answer the following questions about your home to generate your customized estimates for solar installation.")

#Read in most common values
agg_df = pd.read_csv('most_common_values_by_latlong.csv')

# Function for finding nearest lat long
def haversine_distance(user_coordinates, reference_coordinates):
    """
    Calculate the Haversine distance between two points on the Earth specified by their latitude and longitude.
    
    Parameters:
    user_coordinates is a tuple of (lat1, lon1) - latitude and longitude of the first point in decimal degrees
    reference_coordinates is a tuple of (lat2, lon2) - latitude and longitude of the second point in decimal degrees
    
    Returns:
    Distance between the two points in miles.
    """
    lat1 = user_coordinates[0]
    lon1 = user_coordinates[1]
    lat2 = reference_coordinates[0]
    lon2 = reference_coordinates[1]
    
    
    # Radius of the Earth in miles
    R = 3958.8

    # Convert latitude and longitude from degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Difference in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in miles
    distance = R * c

    return distance

# Function for finding default values of lat long
def return_closest_loc_default_values(user_lat_long, default_values_by_latlong_df):
    """
    user_lat_long needs to be a tuple of (lat, long)
    default_values_by_latlong_df needs to have a column called "Lat_long" where each value is a tuple of (lat, long)
    """
    dist_list_miles = []

    for lat_long in default_values_by_latlong_df["Lat_long"]:
        lat = float(lat_long.replace('(', '').replace(')', '').split(', ')[0])
        long = float(lat_long.replace('(', '').replace(')', '').split(', ')[1])
        lat_long = (lat, long)
        dist = haversine_distance(user_lat_long, lat_long)
        dist_list_miles.append(dist)
    
    default_values_by_latlong_df["dist_from_user_loc"] = dist_list_miles
    default_values_by_latlong_df = default_values_by_latlong_df.sort_values(by = ["dist_from_user_loc"], 
                                                                            ascending = True).reset_index(drop = True)
    return default_values_by_latlong_df.head(1)

# Function for predicting solar ghi
def predict_ghi_kWhr(lat, long):
    #Generously sized
    lat_min, lat_max = 24, 50
    long_min, long_max = -130, -65.5

    # Check if the input coordinates are within the continental US
    if not (lat_min <= lat <= lat_max and long_min <= long <= long_max):
        return "Not in continental US"
    
    input_shaped = np.array([[lat, long]]).reshape(1, 1, 2, 1)
    
    predicted_ghi = solar_model.predict(input_shaped)[0][0]
    
    return predicted_ghi


# Function for calculating final estimate
def total_solar_panels(GHI, kwh_residential_usage, roof_area, state, rates_df):
    """
    GHI: annual solar radiation in Wh/m2 for the customer's location
    kWh_residential_usage: Annual electric consumption (kWh) for the customer's home
    state: state of location of customer's home. The state name should be a string in title case, eg. West Virgina, New York, Utah, etc.
    roof_area: roof area of customer's home in sq-ft
    rates_df: avg elec cost (cents per kWh) by state (https://www.electricchoice.com/electricity-prices-by-state/) 
    
    key assumptions:
    1. The customer will be installing solar panels on their whole roof (max possible area).
    2. Solar panel efficiency is based on values from : https://css.umich.edu/publications/factsheets/energy/photovoltaic-energy-factsheet
    3. Operating efficiency involves the solar incidence angle and electrical losses. (Assuming 75% for now. To be updated...)
    4. typical solar panel size is 5.4 ft x 3.25 ft ~ 18 sq-ft (https://us.sunpower.com/solar-resources/how-many-solar-panels-do-you-need-panel-size-and-output-factors)
    """
    solar_panel_efficiency = 0.17
    operating_efficiency = 0.75
    solar_panel_area_sq_ft = 18
    elec_rate = rates_df[rates_df["State"] == state]["Rate_c_per_kWh"].reset_index(drop = True).iloc[0]
    
    # Convert GHI from Wh/m2 to kWh/sq-ft: 1Wh/m2 = 9.2903 x10^-5 kwh/sq-ft
    GHI_kWh_sqft = GHI * 9.2903e-5
    #print(f"GHI_kWh_sqft = {GHI_kWh_sqft}")
    # Annual Solar panel output (kWh/sq-ft) = GHI_kWh_sqft * solar_panel_efficiency
    solar_panel_output_per_sqft = GHI_kWh_sqft * solar_panel_efficiency * operating_efficiency
    #print(f"solar_panel_output_per_sqft = {solar_panel_output_per_sqft}")
    # Annual Output for one solar panel (kWh) = solar_panel_output_per_sqft * solar_panel_area_sq_ft
    solar_panel_output = solar_panel_output_per_sqft * solar_panel_area_sq_ft
    #print(f"solar_panel_output = {solar_panel_output}")
    # Number of solar panels needed to meet annual home energy = kwh_residential_usage/solar_panel_output
    n_solar_panels = int(kwh_residential_usage/solar_panel_output)
    #print(f"n_solar_panels = {n_solar_panels}")
    # Maximum possible solar panels available to install on customer's roof, 
    # leaving 10% roof area unoccupied, and rounding down to avoid fractional solar panels
    max_possible_solar_panels_on_roof = int((roof_area/solar_panel_area_sq_ft) * 0.9)
    #print(f"max_possible_solar_panels_on_roof = {max_possible_solar_panels_on_roof}")
    
    if max_possible_solar_panels_on_roof <= n_solar_panels:
        final_n_solar_panels = max_possible_solar_panels_on_roof
        perc_energy_met = round(max_possible_solar_panels_on_roof*100/n_solar_panels)
    else:
        final_n_solar_panels = n_solar_panels
        perc_energy_met = 100
        
    dollars_saved = round((perc_energy_met/100 * kwh_residential_usage) * elec_rate/100, 2)

    kwh_monthly_prediction = round(kwh_residential_usage/12,2)
        
    return f"We estimated your average monthly electric consumption to be {kwh_monthly_prediction} kWh. You can install {final_n_solar_panels} solar panels on your roof, and it will meet {perc_energy_met}% of your total annual energy requirements! Also, you will save approx ${dollars_saved} per year."


# Initialize session states for address form and default values
if 'address_submitted' not in st.session_state:
   st.session_state.address_submitted = False
if 'occupants_mode' not in st.session_state:
   st.session_state.occupants_mode = ""
if 'sqft_bin_mode' not in st.session_state:
   st.session_state.sqft_bin_mode = ""
if 'vintage_mode' not in st.session_state:
   st.session_state.vintage_mode = ""
if 'heating_mode' not in st.session_state:
   st.session_state.heating_mode = ""
if 'heatingsetpoint_mode' not in st.session_state:
   st.session_state.heatingsetpoint_mode = ""
if 'coolingsetpoint_mode' not in st.session_state:
   st.session_state.coolingsetpoint_mode = ""
if 'geom_building_mode' not in st.session_state:
   st.session_state.geom_building_mode = ""
if 'water_heater_mode' not in st.session_state:
   st.session_state.water_heater_mode = ""
if 'lighting_mode' not in st.session_state:
   st.session_state.lighting_mode = ""
if 'ducts_mode' not in st.session_state:
   st.session_state.ducts_mode = ""
if 'cooling_eff_mode' not in st.session_state:
   st.session_state.cooling_eff_mode = ""
if 'nearest_lat' not in st.session_state:
   st.session_state.nearest_lat = ""
if 'nearest_long' not in st.session_state:
   st.session_state.nearest_long = ""
if 'state' not in st.session_state:
   st.session_state.state = ""
if 'lat' not in st.session_state:
   st.session_state.lat = ""
if 'long' not in st.session_state:
   st.session_state.long = ""

# Set up google maps API 
google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY')

# First form to get address and geocode it
with st.form(key = "address"):
   st.write("**1. Enter your address.**" )
   address = st.text_input('If you don\'t want to provide your address, you can enter your city/state or zip')
   params = {'key' : google_maps_api_key,
'address': address}
   base_url = 'https://maps.googleapis.com/maps/api/geocode/json?'
   # Send to google maps API
   response = requests.get(base_url, params = params).json()
   lat = 0
   # Get lat long
   if response['status'] == 'OK':
      geometry = response['results'][0]['geometry']
      lat = geometry['location']['lat']
      long = geometry['location']['lng']
      latlong = (lat, long)

      # Get state name
      url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{long}&key={google_maps_api_key}"
      response = requests.get(url)
      if response.status_code == 200:
        data = response.json()
        if 'results' in data and len(data['results']) > 0:
            for component in data['results'][0]['address_components']:
                if 'administrative_area_level_1' in component['types']:
                    st.session_state.state = component['long_name']
                    

   # form submit button
   address_submit = st.form_submit_button(label = "Confirm address")
   if address_submit:
      # As long as lat = 0, geocoding call didn't work
      if lat == 0:
         st.markdown("Please enter a valid address")
      else:
         st.session_state.address_submitted = True
         st.write('Address confirmed!')
         
         default_vals = return_closest_loc_default_values(user_lat_long = latlong, default_values_by_latlong_df = agg_df)

         # Set all default values
         st.session_state.lat = lat
         st.session_state.long = long
         st.session_state.nearest_lat = float(default_vals['Lat_long'][0].replace('(', '').replace(')', '').split(',')[0])
         st.session_state.nearest_long = float(default_vals['Lat_long'][0].replace('(', '').replace(')', '').split(',')[1])
         st.session_state.occupants_mode = default_vals['Most_common_in.occupants'][0]
         st.session_state.sqft_bin_mode = default_vals['Most_common_sqft_binned'][0]
         st.session_state.vintage_mode = default_vals['Most_common_in.vintage'][0]
         st.session_state.heating_mode = default_vals['Most_common_hvac_heating_fuel_new'][0]
         st.session_state.heatingsetpoint_mode = default_vals['Most_common_heating_setpoint'][0]
         st.session_state.coolingsetpoint_mode = default_vals['Most_common_cooling_setpoint'][0]
         st.session_state.geom_building_mode = default_vals['Most_common_in.geometry_building_type_height'][0]
         st.session_state.water_heater_mode = default_vals['Most_common_electric_water_heater'][0]
         st.session_state.lighting_mode = default_vals['Most_common_Lighting_new'][0]
         st.session_state.ducts_mode = default_vals['Most_common_in.hvac_has_ducts'][0]
         st.session_state.cooling_eff_mode = default_vals['Most_common_hvac_cooling_eff_new'][0]


# Dictionaries to map user inputs to model input values
occupants_mapping = {1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'10+', 11:'10+', 12:'10+', 13:'10+', 14:'10+', 15:'10+', 16:'10+', 17:'10+', 18:'10+', 19:'10+', 20:'10+'}
sqft_bin_mapping = {'<1,000 sqft':'<1000', '1,000-1,499 sqft':'1000-1499', '1,500-1,999 sqft':'1500-1999', '2,000-2,499 sqft':'2000-2499', '2,500-3,499 sqft':'2500-3499', '3,500+ sqft':'3500+'}
heating_mapping = {'Electric - Heat pump':'Electric-HP', 'Electric - Boiler or furnace':'Electric-resistance', 'No Heating or Non-Electric - Fuel, gas, or propane boiler or furnace':'Non-electric'}
washer_mapping = {'<5 loads per week':'low', '5-6 loads per week':'avg', '7+ loads per week':'high', 'Don\'t have a washing machine':'none'}
sqft_avg_mapping = {'<1,000 sqft':500, '1,000-1,499 sqft':1250, '1,500-1,999 sqft':1750, '2,000-2,499 sqft':2250, '2,500-3,499 sqft':3000, '3,500+ sqft':4500}
water_heater_mapping = {'Electric Resistance Standard': 'Electric Standard', 'Electric Resistance Premium':'Electric Premium', 'Electric Resistance Tankless':'Electric Tankless', 'Electric Heat Pump': 'Electric Heat Pump, 80 gal'}

# Once address confirmed, move onto next questionairre
if st.session_state.address_submitted:
   time.sleep(.5)
   with st.form(key = "questionaire"):
      st.write("**2. Tell us about your home**")

      occupants = st.slider('How many people live in your home?', min_value=1, max_value=15, step = 1, value = st.session_state.occupants_mode)
      occupants_model = occupants_mapping[occupants]

      geom_building = st.selectbox('Is your home attached to another home or detached (on its own)?', ['Attached', 'Detached', 'Not sure'])
      if geom_building == 'Not sure':
         geom_building_model = st.session_state.geom_building_mode
      elif geom_building == 'Attached':
         geom_building_model = "Single-Family Attached"
      else:
         geom_building_model = "Single-Family Detached"

      sqft_bin = st.selectbox('What is the square footage of your home?', ['<1,000 sqft', '1,000-1,499 sqft', '1,500-1,999 sqft', '2,000-2,499 sqft', '2,500-3,499 sqft', '3,500+ sqft', 'Not sure'])
      if sqft_bin == 'Not sure':
         sqft_bin_model = st.session_state.sqft_bin_mode
      else:
         sqft_bin_model = sqft_bin_mapping[sqft_bin]

      floors = st.number_input('How many floors are in your house?', min_value=1, max_value=5, step=1)
      # Use sqft and floors to estimate roof size
      roof_size_sqft = sqft_avg_mapping[sqft_bin]/floors
      # Convert to m^2 for solar model
      roof_size_m2 = roof_size_sqft/10.764

      vintage = st.selectbox('What year was your home built, or when was the last time there were significant changes made to the walls/roof/windows/foundation?', ['<1940','1940s', '1950s', '1960s', '1970s','1980s', '1990s', '2000s', '>2000s', 'Not sure'])
      if vintage == 'Not sure':
         vintage_model = st.session_state.vintage_mode
      elif vintage == ">2000s":
         vintage_model = '2010s'
      else:
         vintage_model = vintage

      washer_usage = st.selectbox('How often does your household use your clothes washing machine for laundry?', ['<5 loads per week', '5-6 loads per week', '7+ loads per week', 'Don\'t have a washing machine'])
      washer_usage_model = washer_mapping[washer_usage]

      #washer_type = st.selectbox('Is your washing machine Energy Star certified?', ['Yes, Energy Star certified', 'No, standard washing machine', 'Don\'t have a washing machine'])
   # According to energy star, average american family does 300 loads of laundry per year = ~6 loads per week

      lighting = st.selectbox('What type of lightbulbs do you have in your home?', ['LED or Fluorescent', 'Incandescent', 'Not sure'])
      if lighting == 'Not sure':
         lighting_model = st.session_state.lighting_mode
      elif lighting == 'LED or Fluorescent':
         lighting_model = 'LED_CFL'
      else:
         lighting_model = 'Incandescent'

      water_heater = st.selectbox('What kind of water heater do you have?', ['Electric Resistance Standard', 'Electric Resistance Premium', 'Electric Resistance Tankless', 'Electric Heat Pump', 'No water heater or non-electric water heater', 'Not sure'])
      if water_heater == 'Not sure':
         water_heater_model = st.session_state.water_heater_mode
      elif water_heater == 'No water heater or non-electric water heater':
         water_heater_model = 'Non-electric Water Heater'
      else:
         water_heater_model = water_heater_mapping[water_heater]

      heating = st.selectbox('What type of heating system do you have?', ['Electric - Heat pump', 'Electric - Boiler or furnace', 'No Heating or Non-Electric - Fuel, gas, or propane boiler or furnace', 'Not sure'])
      if heating == "Not sure":
         heating_model = st.session_state.heating_mode
      else:
         heating_model = heating_mapping[heating]

      cooling_eff = st.selectbox('What efficiency is your HVAC cooling system?', ['avg', 'high', 'low', 'Not sure'])
      if cooling_eff == 'Not sure':
         cooling_eff_model = st.session_state.cooling_eff_mode
      else:
         cooling_eff_model = cooling_eff

      ducts = st.selectbox("Does your HVAC system have ducts?", ['Yes', 'No', 'Not sure'])
      if ducts == 'Not sure':
         ducts_model = st.session_state.ducts_mode
      else:
         ducts_model = ducts

      heatingsetpoint = st.slider('What temperature do you usually turn your heater to in the winter?', min_value=60, max_value=80, step = 1, value = st.session_state.heatingsetpoint_mode)

      coolingsetpoint = st.slider('What temperature do you usually turn your AC or cooling system to in the summer?', min_value=60, max_value=80, step = 1, value = st.session_state.coolingsetpoint_mode)

      submit_qs = st.form_submit_button(label = "Get your estimates")

   # Once questionairre submitted, calculate final estimates
   if submit_qs:
      # Get example data for encoder
      cat_data = pd.read_csv('Example categorical data for encoding.csv')
      # Replace values of relevant columns with user inputs
      cat_data['sqft_binned'] = sqft_bin_model
      cat_data['hvac_heating_fuel_new'] = heating_model
      #cat_data['in.weather_file_latitude'] = st.session_state.nearest_lat
      #cat_data['in.weather_file_longitude'] = st.session_state.nearest_long
      cat_data['in.vintage'] = vintage_model
      cat_data['in.occupants'] = occupants_model
      cat_data['in.geometry_building_type_height'] = geom_building_model
      cat_data['electric_water_heater'] = water_heater_model
      cat_data['clothes_washer_usage_new'] = washer_usage_model
      cat_data['Lighting_new'] = lighting_model
      cat_data['hvac_cooling_eff_new'] = cooling_eff_model
      cat_data['in.hvac_has_ducts'] = ducts_model

      # Load encoder
      encoder = joblib.load('target_encoder.pkl')

      # Encode categorical data
      cat_data_encoded = encoder.transform(cat_data)
      # Select final columns for modeling
      final_cat_cols = ['in.geometry_building_type_height', 'in.hvac_has_ducts', 'in.occupants', 'in.vintage', 'sqft_binned', 'electric_water_heater', 'hvac_heating_fuel_new', 'Lighting_new', 'hvac_cooling_eff_new', 'clothes_washer_usage_new']
      cat_data_encoded_reduced = cat_data_encoded[final_cat_cols]

      # Create numerical dataset
      num_values = [st.session_state.lat, st.session_state.long, coolingsetpoint, heatingsetpoint]
      num_data = pd.DataFrame(columns = ['in.weather_file_latitude', 'in.weather_file_longitude', 'cooling_setpoint', 'heating_setpoint'], data = [num_values])

      # Get final features and filter dataset
      model_data = pd.concat([num_data, cat_data_encoded_reduced], axis = 1)

      # Load model and run predictions
      model = joblib.load('xgb_model.sav')
      kwh_prediction = model.predict(model_data)[0]

      # Load solar model
      solar_model = tf.keras.models.load_model('ghi_prediction_model_updated.keras')
      efficiency_factor = .2 # %conversion efficiency of solar panels to usable power, research shows ~20% for residential panels
      # Confirm formula - based on output of model do we still need to multiple by efficiency factor, 1000, and 365???
      ghi_est = predict_ghi_kWhr(st.session_state.lat, st.session_state.long)

      # Load electric rates by state
      elec_rates_df = pd.read_csv("Elec_rates_by_state.csv")

      final_estimates = total_solar_panels(GHI = ghi_est, 
                   kwh_residential_usage = kwh_prediction, 
                   roof_area = roof_size_sqft,
                   state = st.session_state.state,
                   rates_df = elec_rates_df)

      st.write(f"{final_estimates}")

      with st.form(key = "email"):
         email = st.text_input('Enter your email to save your estimates')
         submit_email = st.form_submit_button(label = "Email estimates")
