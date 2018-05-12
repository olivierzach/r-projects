# https://www.r-bloggers.com/using-rvest-to-scrape-an-html-table/

# phish web scraping function == TOP SHOWS
# example url
# http://phish.net/music/ratings/2016

# source the phish scraping functino
source(phish.scrape)

# define the years you want to scrape
years = as.list(1989:2017)


# phish scraping function
phish.scrape = function(years) {¬}


# run the phish scrape function
phish.scrape(years = years)


# EDA PHISH ---------------------------------------------------------------

phish.analysis. <- unique(phish.analysis) %>% 
        as.data.frame() %>% 
        plyr::rename(c("X..of.Votes" = "Reviews")) %>% 
        filter(Reviews > 11) %>% 
        mutate(count.venue = as.numeric(ave(as.character(Venue), Venue, FUN = length)))

str(phish.analysis.); summary(phish.analysis.)


epoch <- phish.analysis. %>% 
        as.data.frame() %>% 
        dplyr::select(Epoch, Rating) %>% 
        group_by(Epoch) %>% 
        summarise(show.score = mean(Rating))

ggplot(data = epoch, aes(x = Epoch, y = show.score)) +
        geom_point(stat = 'identity')

season <- phish.analysis. %>% 
        as.data.frame() %>% 
        dplyr::select(Tour.Season, Rating) %>% 
        group_by(Tour.Season) %>% 
        summarise(show.score = mean(Rating))

ggplot(data = season, aes(x = Tour.Season, y = show.score)) +
        geom_bar(stat = 'identity')


ggplot(data = phish.analysis.) +
        geom_histogram(aes(x = Rating), binwidth = .5)


ggplot(data = phish.analysis.) +
        geom_histogram(aes(x = X..of.Votes), binwidth = 50)



ggplot(data = phish.analysis., aes(x = Rating, color = Epoch)) +
        geom_freqpoly(binwidth = .5)


ggplot(data = phish.analysis., aes(x = Rating, color = Tour.Season, y = ..density..)) +
        geom_freqpoly(binwidth = .5)


ggplot(data = phish.analysis., aes(x= Rating, y = ..density..)) +
        geom_freqpoly(aes(color = Epoch), binwidth = .5)


ggplot(data = phish.analysis., aes(x = reorder(Epoch, Rating, FUN = median), y = Rating)) +
        geom_boxplot(varwidth = T) +
        geom_jitter(pch = 22, aes(color = Tour.Season), alpha = 1/3) +
        theme_fivethirtyeight()


ggplot(data = phish.analysis., aes(x = reorder(Tour.Season, Rating, FUN = median), y = Rating)) +
        geom_boxplot(varwidth = T) +
        geom_jitter(aes(color = Epoch), pch = 22, alpha = 1) +
        labs(y = "") +
        theme_fivethirtyeight()


ggplot(data = phish.analysis., aes(x = reorder(Last.Album, Rating, FUN = median), y = Rating, color = Epoch)) +
        geom_boxplot(varwidth = T) +
        coord_flip() +
        labs(x = "")

install.packages("ggthemes")
library(ggthemes)

ggplot(data = phish.analysis., aes(x = reorder(Show.Year, Rating, FUN = median), y = Rating, color = Epoch)) +
        geom_boxplot(varwidth = T) +
        coord_flip() +
        labs(x = "", title = "Highest Rated Phish Shows by Year", subtitle = "Median Show Rating") +
        theme_fivethirtyeight()


ggplot(data = phish.analysis., aes(x = reorder(Last.Album, Rating, FUN = median), y = Rating, color = Epoch)) +
        geom_boxplot(varwidth = T) +
        coord_flip() +
        labs(x = "Rating", y = "Rating", title = "Show Ratings by Album Released before Tour", subtitle = "Step into the Freezer...") +
        theme_fivethirtyeight()


ggplot(data = phish.analysis., aes(x = reorder(State, Rating, FUN = median), y = Rating)) +
        geom_boxplot(varwidth = T) +
        coord_flip() +
        theme_solarized() +
        labs(x = "")


ggplot(data = phish.analysis., aes( x = X..of.Votes, y = Rating, color = Epoch)) +
        geom_point() +
        geom_smooth(se = F)



ggplot(data = phish.analysis., aes(x = Show.Date, y = Rating, color = Epoch)) +
        geom_line() +
        geom_smooth(se = T)


ggplot(data = phish.analysis. %>% filter(Epoch == "1.0"), aes(x = Show.Date, y = Rating, color = Epoch)) +
        geom_line() +
        geom_smooth(se = T) 


ggplot(data = phish.analysis. %>% filter(Epoch == "2.0"), aes(x = Show.Date, y = Rating, color = Epoch)) +
        geom_line() +
        geom_smooth(se = T) 


ggplot(data = phish.analysis. %>% filter(Epoch == "3.0"), aes(x = Show.Date, y = Rating, color = Epoch)) +
        geom_line() +
        geom_smooth(se = T) 





# PHISH MODEL -------------------------------------------------------------

library(caret);library(gbm);library(plyr);library(dplyr)


set.seed(3433)

phish.model <- phish.analysis. %>% 
        dplyr::select(., -Rating, -Reviews, -Country, -Venue, -Show.Year, -Show.Month, -Show.Day , -count.venue) %>% 
        as.data.frame() %>% 
        mutate(Rating.y = log(Rating.y)) %>% 
        mutate_if(is.character, as.factor)

hist(phish.model$Rating.y); str(phish.model)

inTrain <- createDataPartition(phish.model$Rating.y, p = .7,  list = F)

training <- phish.model[inTrain,]
testing <-  phish.model[-inTrain,]

str(training)
dim(training)

str(testing)
dim(testing)


set.seed(62433)

# logistic regression fit
log.fit <- train(Rating.y ~ ., data = training, method = "glm")

summary(log.fit$finalModel)

exp(coef(log.fit $finalModel))

pred.log <- predict(log.fit, newdata = testing)

residuals.log <- cbind(pred.log , testing$Rating.y) %>% 
        as_tibble() %>% 
        mutate(pred = exp(pred.log ),
               obs = exp(testing$Rating.y),
               resid = (pred - obs)^2,
               MAE = mean(resid),
               RMSE = sqrt(mean(resid^2)),
               pred.score = pred * 5,
               obs.score = obs * 5,
               res.score = (pred.score - obs.score)^2,
               MAE.score = mean(res.score),
               RMSE.score = sqrt(mean(res.score^2))) %>% 
        cbind(., testing) %>% 
        as_tibble()

str(residuals.log ); View(residuals.log )

ggplot(residuals.log, aes(x = obs, y = resid)) +
        geom_point()




# forest boosted fit
mod.gbm <- train(Rating.y ~ ., data = training,  method = "gbm")

summary(mod.gbm)

# boosted predict
pred.gbm <- predict(mod.gbm, testing)

residuals.gbm <- cbind(pred.gbm, testing$Rating.y) %>% 
        as_tibble() %>% 
        mutate(pred = exp(pred.gbm),
               obs = exp(testing$Rating.y),
               resid = (pred - obs)^2,
               MAE = mean(resid),
               RMSE = sqrt(mean(resid^2)),
               pred.score = pred * 5,
               obs.score = obs * 5,
               res.score = (pred.score - obs.score)^2,
               MAE.score = mean(res.score),
               RMSE.score = sqrt(mean(res.score^2))) %>% 
        cbind(., testing) %>% 
        as_tibble()

str(residuals.gbm); View(residuals.gbm)

ggplot(residuals.gbm , aes(x = obs, y = resid)) +
        geom_point()


# random forest fit
mod.tree <- train(Rating.y ~ ., data = training, method = "rf")

mod.tree$finalModel

summary(mod.tree$finalModel)

# random forest predict
pred.tree <- predict(mod.tree, testing)


residuals.rf <- cbind(pred.tree, testing$Rating.y) %>% 
        as_tibble() %>% 
        mutate(pred = exp(pred.tree),
               obs = exp(testing$Rating.y),
               resid = (pred - obs)^2,
               MAE = mean(resid),
               RMSE = sqrt(mean(resid^2)),
               pred.score = pred * 5,
               obs.score = obs * 5,
               res.score = (pred.score - obs.score)^2,
               MAE.score = mean(res.score),
               RMSE.score = sqrt(mean(res.score^2))) %>% 
        cbind(., testing) %>% 
        as_tibble()

str(residuals.rf ); View(residuals.rf )

ggplot(residuals.rf  , aes(x = obs, y = resid)) +
        geom_point()






# linear discriminant model fit
mod.glmnet <- train(Rating.y ~ ., data = training, method = "glmnet")

# linear discriminant predict
pred.glmnet <- predict(mod.glmnet , testing)

residuals.net <- cbind(pred.glmnet, testing$Rating.y) %>% 
        as_tibble() %>% 
        mutate(pred = exp(pred.glmnet),
               obs = exp(testing$Rating.y),
               resid = (pred - obs)^2,
               MAE = mean(resid),
               RMSE = sqrt(mean(resid^2)),
               pred.score = pred * 5,
               obs.score = obs * 5,
               res.score = (pred.score - obs.score)^2,
               MAE.score = mean(res.score),
               RMSE.score = sqrt(mean(res.score^2))) %>% 
        cbind(., testing) %>% 
        as_tibble()

str(residuals.net); View(residuals.net)

ggplot(residuals.net  , aes(x = obs, y = resid)) +
        geom_point()



# linear discriminant model fit
mod.treebag <- train(Rating.y ~ ., data = training, method = "treebag")

# linear discriminant predict
pred.treebag <- predict(mod.treebag, testing)

residuals.treebag <- cbind(pred.treebag , testing$Rating.y) %>% 
        as_tibble() %>% 
        mutate(pred = exp(pred.treebag),
               obs = exp(testing$Rating.y),
               resid = (pred - obs)^2,
               MAE = mean(resid),
               RMSE = sqrt(mean(resid^2)),
               pred.score = pred * 5,
               obs.score = obs * 5,
               res.score = (pred.score - obs.score)^2,
               MAE.score = mean(res.score),
               RMSE.score = sqrt(mean(res.score^2))) %>% 
        cbind(., testing) %>% 
        as_tibble()

str(residuals.treebag ); View(residuals.treebag)

ggplot(residuals.net) +
        geom_jitter(aes(x = Show.Date, y = obs.score), color = "black", alpha = .4) +
        geom_jitter(aes(x = Show.Date,y = pred.score), color = "red", alpha = .4) +
        geom_smooth(aes(x = Show.Date, y = obs.score), color = "black") +
        geom_smooth(aes(x = Show.Date, y = pred.score), color = "red")



model.scorecard <- cbind(residuals.gbm$MAE.score, residuals.treebag$MAE.score, 
                         residuals.rf$MAE.score, residuals.net$MAE.score,
                         residuals.log$MAE.score) %>% 
        as_tibble() %>% 
        plyr::rename(c("V1" = "gbm", "V2" = "treebag", "V3" =  "rforest", "V4" =  "glmnet", "V5" = "logreg")) %>% 
        unique() %>% 
        mutate_all(funs(round(.,4)*100))

View(model.scorecard); str(model.scorecard)



# build summer 2018 prediction dates:

# 7/17 – Lake Tahoe Outdoor Arena at Harveys, Stateline, NV
# 7/18 – Lake Tahoe Outdoor Arena at Harveys, Stateline, NV
# 7/20 – The Gorge Amphitheatre, George, WA
# 7/21 – The Gorge Amphitheatre, George, WA
# 7/22 – The Gorge Amphitheatre, George, WA
# 7/24 – Bill Graham Civic Auditorium, San Francisco, CA
# 7/25 – Bill Graham Civic Auditorium, San Francisco, CA
# 7/27 – The Forum, Inglewood, CA
# 7/28 – The Forum, Inglewood, CA
# 7/31 – Austin360 Amphitheater, Austin, TX
# 8/03 – Verizon Amphitheatre Alpharetta, GA
# 8/04 – Verizon Amphitheatre Alpharetta, GA
# 8/05 – Verizon Amphitheatre Alpharetta, GA
# 8/07 – BB&T Pavilion, Camden, NJ
# 8/08 – BB&T Pavilion, Camden, NJ
# 8/10 – Coastal Credit Union Music Park at Walnut Creek, Raleigh, NC
# 8/11 – Merriweather Post Pavilion, Columbia, MD
# 8/12 – Merriweather Post Pavilion, Columbia, MD
# 8/31 – Dick’s Sporting Goods Park, Commerce City, CO
# 9/01 – Dick’s Sporting Goods Park, Commerce City, CO
# 9/02 – Dick’s Sporting Goods Park, Commerce City, CO

summer.18 <- tibble(
        ~Show.Date, ~City, ~State, ~Tour.Season, ~Holiday.Run, ~Holiday.Set, ~Epoch, ~Last.Album,
        
)



# ensemble method data frame
comb.df = data.frame(pred.rf, pred.gbm, pred.lda, diagnosis = testing$diagnosis)

# ensemble fit
mod.ens <- train(diagnosis ~ ., method = "rf", data = comb.df)

# ensemble predict
pred.ens <- predict(mod.ens, comb.df)

# ensemble fit accuracy
confusionMatrix(pred.ens, testing$diagnosis)$overall[1]
# Accuracy 
# 0.8170732 





















# phish web scraping = setlist data
# example url
# http://phish.net/setlists/phish-december-31-2017-madison-square-garden-new-york-ny-usa.html
# http://phish.net/setlists/phish-april-20-1989-humphries-house-the-zoo-amherst-college-amherst-ma-usa.html

# manipulate phish.analysis data to get setlist urls
phish.dates = phish.analysis %>% 
        dplyr::select(Show.Date, Show.Day, Show.Month, Show.Year,Country, Venue, City, State) %>% 
        mutate(month.name = tolower(format(ISOdate(year = Show.Year, month = Show.Month, day = Show.Day),
                                   "%B")),
               venue.sub = tolower(gsub(" ", "-", Venue, fixed = T)),
               city.sub = tolower(gsub(" ","-", City, fixed = T)), 
               url = paste("http://phish.net/setlists/phish-",paste(month.name),
                           "-",paste(Show.Day),"-",paste(Show.Year),"-",
                           paste(venue.sub),"-",paste(city.sub),
                           "-",paste(tolower(State)),"-",
                           paste(tolower(Country)),".html"),
               url.sub = gsub(" ","", url, fixed = T),
               url.sub = gsub("(","",url.sub, fixed = T),
               url.sub = gsub(")","", url.sub,fixed = T ),
               url.sub = gsub("'","s",url.sub, fixed = T),
               url.sub = gsub("--mexico","-mexico", url.sub, fixed = T),
               url.sub = gsub(",","",url.sub, fixed = T)) %>% 
        filter(Show.Year == "2017", Country == "USA") %>% 
        dplyr::select(url.sub) %>% 
        as.data.frame()
               
str(phish.dates)
unique(phish.dates$venue.sub)
unique(phish.dates$url.sub)
length(phish.dates)
View(phish.dates)

phish.setlists = NULL

for (p in 1:nrow(phish.dates)) {
        
        url = paste(phish.dates[p,])
        
        phish.read = url %>%
                read_html() %>% 
                html_nodes(xpath='/html/body/div[1]/div[2]/div[1]/div[2]/div[2]') %>% 
                html_text() %>% 
                gsub('[\r\n\t]', '', .)

        phish.df = plyr::rbind.fill(phish.setlists, phish.read)
        
}



# let's scrape the phish.in dataset for all shows by year
# https://www.r-bloggers.com/accessing-apis-from-r-and-a-little-r-programming/






# /html/body/div[1]/div[2]/div[1]


# references
# # test case = download top ranked shows from 1989
# url <- "http://phish.net/music/ratings/1989"
# 
# phish.89 <- url %>%
#   read_html() %>%
#   html_nodes(xpath='/html/body/div[1]/div[2]/div[1]/table') %>%
#   html_table()
# phish.89  <- phish.89 [[1]]
# 
# head(phish.89)

# FOO <- function(data, nSubsets, nSkip){
#         outList <- vector("list", length = nSubsets)
#         totRow <- nrow(data)
#         
#         for (i in seq_len(nSubsets)) {
#                 rowsToGrab <- seq(i, totRow, nSkip)
#                 outList[[i]] <- data[rowsToGrab ,] 
#         }
#         return(outList)
#         
#         
# }
# 
# 
# 
# #http://phish.net/music/ratings
# install.packages("XML")
# library(XML) 
# 
# URL <- "http://phish.net/music/ratings/1989" 
# root <- xmlTreeParse(URL, useInternalNodes = TRUE) 
# 
# fn <- function(node) { 
#         id <- xmlAttrs(node)["id"] 
#         parent.id <- xmlAttrs(xmlParent(node))["id"] 
#         setNames(head(c(id, parent.id, NA), 2), c("id", "parent")) 
# } 
# 
# parents <- t(xpathSApply(root, "//component", fn)) 
# 
# parents[1:4, ]
# 
# 
# 
# 
# readHTMLTable = function(tb)
#         {
#                 # get the header information.
#                 colNames = sapply(tb[["thead"]][["tr"]]["th"], xmlValue)
#                 vals = sapply(tb[["tbody"]]["tr"],  function(x) sapply(x["td"], xmlValue))
#                 matrix(as.numeric(vals[-1,]),
#                        nrow = ncol(vals),
#                        dimnames = list(vals[1,], colNames[-1]),
#                        byrow = TRUE
#                 )
#         }
# 
# 
# readHTMLTable("http://phish.net/music/ratings/1989.html")
# 
# 
# phishdat4[,1:2] <- sapply(phishdat4[,1:2],as.numeric)
#  summary(phishdat4)
#  
#  
#  phish5 <- transform(phishdat4,as.numeric(phishdat4$NULL.Rating),as.numeric(phishdat4$NULL...of.Votes),as.character(phishdat4$NULL.Venue))
# 
#  function(json) {
#          document.write("<div id=\"pnetsetlist\">")
#    
#          for(i =  0, i<json.response.data.length,i++) {
#                  var n = json.response.data[i];
#                  document.write("<h3>"+n.short_date+" <a href=\"http://phish.net/setlists/"+n.permalink+".html\">"+n.venue+"</a></h3><p class='pnetsl'>"+n.setlistdata+"</p>");
#                  if(n.setlistnotes.replace(/^\s+|\s+$/g,"")!='') { document.write("<p class='pnetsn' style=\"font-size:small\">" + n.setlistnotes + "</p>"); }
#          }
#          document.write("</div>"); }
 
 
 
 
 
 