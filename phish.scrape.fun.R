phish.scrape = function(years) {
        
        library(rvest)
        library(plyr)
        library(dplyr)
        library(lubridate)
        
        phish.df = NULL
        
        for (p in 1:length(years)) {
                
                url = paste("http://phish.net/music/ratings/",years[p], sep = "")
                
                phish.read = url %>%
                        read_html() %>% 
                        html_nodes(xpath='/html/body/div[1]/div[2]/div[1]/table') %>% 
                        html_table()
                
                phish.read = phish.read[[1]]
                
                phish.df = plyr::rbind.fill(phish.df, phish.read)
                
        }
        
        
        
        phish.df. = phish.df %>% 
                data.frame() %>% 
                dplyr::mutate(Show.Date   = as.Date(Show.Date),
                              Show.Year   = lubridate::year(Show.Date),
                              Show.Month  = lubridate::month(Show.Date),
                              Show.Day    = lubridate::day(Show.Date),
                              Rating.y    = Rating / 5,
                              Tour.Season = ifelse(Show.Month %in% c(9,10,11,12), "Fall Tour", 
                                                   ifelse(Show.Month %in% c(7,8,9),"Summer Tour",
                                                          ifelse(Show.Month %in% c(4,5,6), "Spring Tour", "Winter Tour"))),
                              Holiday.Run = ifelse(Show.Month == 10 & Show.Day %in% c(28,29,30,31),"Halloween Run",
                                                   ifelse(Show.Month == 12 & Show.Day %in% c(28,29,30,31),"NYE Run","Run")),
                              Holiday.Set = ifelse(Show.Month == 10 & Show.Day == 31,"Costume Set",
                                                   ifelse(Show.Month == 12 & Show.Day == 31,"NYE Set","Normal Set")),
                              Epoch       = ifelse(Show.Date < as.Date("2000-01-01"), "1.0",
                                                   ifelse(Show.Date >= as.Date("2000-01-01") & Show.Date <= as.Date("2005-01-01"), "2.0", "3.0")),
                              Last.Album  = ifelse(Show.Date < as.Date("1989-08-05"), "The White Tape",
                                                   ifelse(Show.Date >= as.Date("1989-08-05") & Show.Date < as.Date("1990-09-21"),"Junta",
                                                          ifelse(Show.Date >= as.Date("1990-09-21") & Show.Date < as.Date("1992-02-12"), "Lawn Boy",
                                                                 ifelse(Show.Date >= as.Date("1992-02-12") & Show.Date < as.Date("1993-02-02"), "A Picture of Nectar",
                                                                        ifelse(Show.Date >= as.Date("1993-02-02") & Show.Date < as.Date("1994-03-29"), "Rift",
                                                                               ifelse(Show.Date >= as.Date("1994-03-29") & Show.Date < as.Date("1996-10-15"), "Hoist",
                                                                                      ifelse(Show.Date >= as.Date("1996-10-15") & Show.Date < as.Date("1998-10-27"), "Billy Breathes",
                                                                                             ifelse(Show.Date >= as.Date("1998-10-27") & Show.Date < as.Date("2000-05-16"), "The Story of the Ghost",
                                                                                                    ifelse(Show.Date >= as.Date("2000-05-16") & Show.Date < as.Date("2002-12-10"), "Farmhouse",
                                                                                                           ifelse(Show.Date >= as.Date("2002-12-10") & Show.Date < as.Date("2004-07-15"), "Round Room",
                                                                                                                  ifelse(Show.Date >= as.Date("2004-07-15") & Show.Date < as.Date("2009-09-08"), "Undermind",
                                                                                                                         ifelse(Show.Date >= as.Date("2009-09-08") & Show.Date < as.Date("2014-06-24"), "Joy",
                                                                                                                                ifelse(Show.Date >= as.Date("2014-06-24") & Show.Date < as.Date("2016-10-07"), "Fuego", "Big Boat")))))))))))))) %>% 
                as.data.frame()
        
        phish.analysis <<- unique(phish.df.)
        
}