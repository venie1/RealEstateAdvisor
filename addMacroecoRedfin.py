import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import requests 

#get the ACS data
acsData = {}

years = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']#, '2024', '2025']
metrics = ["B19013_001E","B17001_002E","B15003_022E", "B23025_005E","B01003_001E"]

for year in years:
  BASE_URL = "https://api.census.gov/data/" + year + "/acs/acs5"


  VARIABLES = ["NAME", "B19013_001E", #median household income
                       "B17001_002E", #no people living in poverty 
                       "B15003_022E", #no people over 25, who earned a bachelor's degree or higher
                       "B23025_005E", #no people over 16 who are unemployed
                       "B01003_001E"] #total population (estimate)
                       #"B27010_017E",] #no people witout health insurence] 

  params = {"get": ",".join(VARIABLES),"for": "county:*","key": "921a84b5ae6780f278b58176b867c14eaf006109"}

  response = requests.get(BASE_URL, params=params)

 
  acsData[year] = {}
  for metric in metrics:
    acsData[year][metric] = {}
  
  if response.status_code == 200:
    data = response.json()
    #print(data)
    #['NAME',                 'B19013_001E', 'B17001_002E', 'B15003_022E', 'B23025_005E', 'B27010_017E', 'state', 'county']
    #['Autauga County, Alabama', '69841',         '6275',      '6518',         '688',         '401', '01', '001']
    for row in data:

      acsData[year]["B19013_001E"][row[0]] = row[1]
      acsData[year]["B17001_002E"][row[0]] = row[2]
      acsData[year]["B15003_022E"][row[0]] = row[3]
      acsData[year]["B23025_005E"][row[0]] = row[4]
      acsData[year]["B01003_001E"][row[0]] = row[5]

    county = "Hillsborough County, Florida"
    print(year + " - - - - -" +  acsData[year]["B19013_001E"][county] + " - " + acsData[year]["B17001_002E"][county] + " - " + acsData[year]["B15003_022E"][county] + " - " + acsData[year]["B23025_005E"][county] + " - " + acsData[year]["B01003_001E"][county] )
  else:
    print("Error:", response.status_code, response.text)

#load the Redfin county data, and start adding the above data
data = []
#open the csv file
with open("county_market_tracker.tsv000") as fd:

    rd = csv.reader(fd, delimiter="\t", quotechar='"')
    for row in rd:
        data.append(row)

#years = []
zcounter = 0
totalcounter = 0
data[0].append("county_w_full_state_name")
for metric in metrics:
    data[0].append(metric)
for row in data[1:]:
    totalcounter += 1

    rows = ["period_begin","period_end","period_duration","region_type","region_type_id","table_id","is_seasonally_adjusted","region","city","state","state_code"
            ,"property_type","property_type_id","median_sale_price","median_sale_price_mom","median_sale_price_yoy","median_list_price","median_list_price_mom","median_list_price_yoy","median_ppsf","median_ppsf_mom","median_ppsf_yoy","median_list_ppsf","median_list_ppsf_mom","median_list_ppsf_yoy","homes_sold","homes_sold_mom","homes_sold_yoy","pending_sales","pending_sales_mom","pending_sales_yoy","new_listings","new_listings_mom","new_listings_yoy","inventory","inventory_mom","inventory_yoy","months_of_supply","months_of_supply_mom","months_of_supply_yoy","median_dom","median_dom_mom","median_dom_yoy","avg_sale_to_list","avg_sale_to_list_mom","avg_sale_to_list_yoy","sold_above_list","sold_above_list_mom","sold_above_list_yoy","price_drops","price_drops_mom","price_drops_yoy","off_market_in_two_weeks","off_market_in_two_weeks_mom","off_market_in_two_weeks_yoy","parent_metro_region","parent_metro_region_metro_code","last_updated"]


    year = row[0][:4]
    #if year not in years:
    #    years.append(year)
    #row.append(year)
    if year == "2024" or year == "2025":
       year = "2023"

    #change state abbreviation to full name in county
    new_county_lookup = row[7].replace(row[10], row[9])
    row.append(new_county_lookup)

    zmarker = False

    #loop through each metric..
    for metric in metrics:
        #if the metric value exists for the given year and county, append the value. if not append a 0
        if new_county_lookup in acsData[year][metric]:
            #if new_county_lookup == "Petersburg Borough, Alaska":
                #print(year + " " + acsData[year][metric][new_county_lookup])
            row.append(acsData[year][metric][new_county_lookup])
        else:
            row.append("")
            zmarker = True
    if zmarker:
        zcounter += 1

#years.sort()
#print(years)


print(" - - - - - - ")
print(totalcounter)
print(zcounter)



    #with open("test.csv", "w", encoding="utf-8", newline="") as out_fd
wr = csv.writer(open("test4.tsv000", "w"), delimiter="\t", quotechar='"', quoting=csv.QUOTE_MINIMAL)


for row in data:
    wr.writerow(row)
print("csv file saved")



