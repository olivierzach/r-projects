
# Orange Country Hospitals --------------------------------------------------------------------

# map the orange country hospitals using lat lon
# use leaflet to show results
# need to find list of oc hospital coordinates


# workflow ------------------------------------------------------------------------------------

# load libraries
pacman::p_load(
  tidyverse, rio, readxl, data.table, acs, tigris, leaflet, scales, raster, htmlwidgets, ggthemes,
  plotly
)

# options and wd
setwd("//mnacpfs01/GPI-DS/Zach Olivier/Dealer Network/")



# get data ------------------------------------------------------------------------------------


(oc_hosp <- tribble(
  ~Hospital, ~Address, ~City, ~State, ~Zip,  ~Website, ~LAT, ~LON,
  'Mission Hospital','27700 Medical Center Rd.', 'Mission Viejo', 'CA', '92691',  'http://www.mission4health.com', 33.561015, -117.665394,
  'Orange Coast Memorial Medical Centerl','9920 Talbert Avenue', 'Fountain Valley', 'CA', '92708',  'www.memorialcare.org/orange_coast', 33.700391, -117.955623,
  'Anaheim Regional Medical Center','1111 W. La Palma Ave.', 'Anahiem', 'CA', '92804',  'http://www.anaheimregionalmc.com/', 33.848175, -117.934526,
  "Children's Hospital of Orange County (CHOC)",'455 South Main Street', 'Orange', 'CA', '92868',  'https://www.choc.org/', 33.780908, -117.866630,
  'Fountain Valley Regional Hospital and Medical Center','17100 Euclid at Warner', 'Fountain Valley', 'CA', '92708',  'https://www.fountainvalleyhospital.com/', 33.716314, -117.937041,
  'Garden Grove Hospital and Medical Center','12601 Garden Grove Blvd', 'Garden Grove', 'CA', '92843',  'https://www.gardengrovehospital.com/', 33.775007, -117.912908,
  'Hoag Memorial Hospital Presbyterian','One Hoag Drive', 'Newport Beach', 'CA', '92658',  'https://www.hoag.org/', 33.624743, -117.930047,
  'Huntington Beach Hospital','17772 Beach Blvd.', 'Huntington Beach', 'CA', '92647',  'https://www.hbhospital.org/', 33.703640, -117.986656,
  'Irvine Regional Hospital and Medical Center','16200 Sand Canyon Avenue ', 'Irvine', 'CA', '92618',  'http://www.ucirvinehealth.org/', 33.660804, -117.772452,
  'Los Alamitos Medical Center ','3751 Katella Avenue', 'Los Alamitos', 'CA', '90720',  'https://www.losalamitosmedctr.com/', 33.804525, -118.066999,
  'Saddleback Memorial Medical Center','24451 Health Center Drive', 'Laguna Hills', 'CA', '92653',  'http://www.saddlebackmemorialmedicalcenter.com/about', 33.6089015, -117.708725,
  'Saddleback Memorial San Clemente Campus','654 Camino de los Mares', 'San Clemete', 'CA', '92673',  'http://www.saddlebackmemorialmedicalcenter.com/about', 33.500419, -117.741466,
  'South Coast Medical Center','31872 Coast Highway', 'Laguna Beach', 'CA', '92651',  'https://southcoast-gmc.com/', 33.778914, -117.865026,
  'St. Joseph Hospital','1100 W Stewart Dr', 'Orange', 'CA', '92868',  'https://www.sjo.org/', 33.782044, -117.865425,
  'St. Jude Medical Center','101 E Valencia Mesa Dr', 'Fullerton', 'CA', '92835',  'https://www.stjude.org/', 33.894345, -117.927220,
  'UCI Medical Center','101 The City Drive South', 'Orange', 'CA', '92868',  'http://www.ucirvinehealth.org/', 33.786112, -117.887980
))




# map -----------------------------------------------------------------------------------------


# make potential new location a special icon
icon.ion <- makeAwesomeIcon(
  icon = 'home',
  markerColor = 'dark blue',
  iconColor = 'white'
)



# plot leaflet map
# werner specfic sales into census tracts with competitive comparisonsations
map_oc_hosp <-leaflet() %>%
   addAwesomeMarkers(
     data = oc_hosp,
     ~LON,
     ~LAT,
     popup = as.character(paste(oc_hosp$Hospital)),
     label = as.character(paste(oc_hosp$Hospital)),
     icon = icon.ion
   ) %>% 
   addProviderTiles(providers$OpenStreetMap) %>%
   addProviderTiles("CartoDB.Positron") %>%
   addTiles(group = "OpenStreetMap")  %>%
   addProviderTiles("Hydda.Full", group = "Full")  %>%
   addProviderTiles("Stamen.Toner", group = "Toner")  %>%
   addProviderTiles("Esri.WorldStreetMap", group = "WorldStreetMap") %>% 
   addLayersControl(
     baseGroups = c(
       "CartoDB.Positron",
       "OpenStreetMap",
       "Full", 
       "Toner",
       "WorldStreetMap"
     ), 
     overlayGroups = c("Markers", "Outline"),
     position = "topleft"
   )

save(map_oc_hosp, file = 'oc_hosp.Rda')

saveWidget(
  map_oc_hosp,
  file="oc_hosp.html",
  selfcontained=T
) 
