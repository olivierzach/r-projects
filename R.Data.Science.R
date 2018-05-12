## R for Data Science

## Part 1: Explore

# goal: get basic toolset of data exploration as soon as possible
# data exploration = looking at data, generating hypothesis, test them, and iterate
# goal is to generate leads and dive into depth later

# explore workflow:
# import data > tidy > transform > visualize > model > go back to transform







# chapter 1 ggplot2 -------------------------------------------------------



## Chapter 1: ggplot2
# ggplot2 is most elegant and versitile = grammar of graphics = coherent system for building graphs

#load the required results
install.packages("tidyverse")
library(ggplot2)
library(plyr)
library(dplyr)


# load the mpg dataset for plotting
mpg = ggplot2::mpg

# create a plot of displacement and highway mpg:
# results show a negative relationship between highway mpg and displacement
p1 = ggplot(data = mpg) +
        geom_point(mapping = aes(x = displ, y = hwy))

p1

# general ggplot structure:

# ggplot(data = <DATA>) +
#         <GEOM_FUNCTION>(mapping = aes(<MAPPINGS>))

# Exercise - plot is blank we have do GEOM_FUNCTION agruement
ggplot(data = mpg)

# Exercise - how many variables are in mtcars? how many columns?
str(mtcars)

# 'data.frame':	32 obs. of  11 variables:
#         $ mpg : num  21 21 22.8 21.4 18.7 18.1 14.3 24.4 22.8 19.2 ...
# $ cyl : num  6 6 4 6 8 6 8 4 4 6 ...
# $ disp: num  160 160 108 258 360 ...
# $ hp  : num  110 110 93 110 175 105 245 62 95 123 ...
# $ drat: num  3.9 3.9 3.85 3.08 3.15 2.76 3.21 3.69 3.92 3.92 ...
# $ wt  : num  2.62 2.88 2.32 3.21 3.44 ...
# $ qsec: num  16.5 17 18.6 19.4 17 ...
# $ vs  : num  0 0 1 1 0 1 0 1 1 1 ...
# $ am  : num  1 1 1 0 0 0 0 0 0 0 ...
# $ gear: num  4 4 4 3 3 3 3 4 4 4 ...
# $ carb: num  4 4 1 1 2 1 4 2 2 4 ...

# make a scatterplot of hwy vs cyl
p2 = ggplot(data = mpg) +
        geom_point(aes(x = hwy, y = cyl))
p2

# what happens when you plot class vs. drv? = categorical variables on both axis

p3 = ggplot(data = mpg) +
        geom_point(aes(x = class, y = drv))
p3

# aesthetic mappings - you can add more variables to you plots in ggplot
# aesthetics are visual properties of your graphs
# example

# class mapped to color
p4 = ggplot(data = mpg) +
        geom_point(aes(x = displ, y = hwy, color = class))
p4

# class mapped to size
p5 = ggplot(data = mpg) +
        geom_point(aes(x = displ, y = hwy, size = class))
p5

# class mapped to alpha = transparency
p6 = ggplot(data = mpg) +
        geom_point(aes(x = displ, y = hwy, alpha = class))
p6

# class mapped to shape = shape of the actual points
p7 = ggplot(data = mpg) +
        geom_point(aes(x = displ, y = hwy, shape = class))
p7


# you can set up aesthetics manually as well
# to set the aesthetic manually == SET THE COMMAND OUTSIDE OF THE AES() AGRUEMENT !!!
# this can be set for color, size, alpha and shape!

p8 = ggplot(data = mpg) +
        geom_point(aes(x = displ, y = hwy), color = "blue")
p8

# using facets to display multiple categorical data
# use the facet_wrap function and facet_grid functions
# facet_wrap / facet_grid takes a mapping "formula" - see examples below

# builds facets on one variable
p9 = ggplot(data = mpg) +
        geom_point(aes(x = displ, y = hwy))+
        facet_wrap(~ class, nrow = 2)
p9

# build facets on multiple variables = facet_grid

p10 = ggplot(data = mpg) +
        geom_point(aes(x = displ, y = hwy)) +
        facet_grid(drv ~ cyl)
p10


## geometric objects
# this is the geom_"""" function in your call to ggplot
# you can get different type plots based on which geom_function you call
# every geom_function takes a mapping arguement
# not all functions can take the same aesthetics

# geom_smooth
p11 = ggplot(data = mpg) +
        geom_smooth(aes(x = displ, y = hwy))
p11


# geom_smooth with linetypes based on "class"
# also includes the geom_point scatterplot of the actual data

p12 = ggplot(data = mpg) +
        geom_point(aes(x = displ, y = hwy, color = drv), show.legend = F) +
        geom_smooth(aes(x = displ, y = hwy, linetype = drv, color = drv), show.legend = F)
p12


# set "global" mappings in the first statement 
# geom function will apply these mappings to any geom_function you call

ggplot(data = mpg, aes(x = displ, y = hwy)) +
        geom_point()+
        geom_smooth()

# "inner" mappings will be displayed first, then the "global" mappings

ggplot(data = mpg, aes(x = displ, y = hwy)) +
        geom_point(aes(color = class)) +
        geom_smooth()

# you can use this idea to map different data to each geom_function call

ggplot(data = mpg, aes(x = displ, y = hwy)) +
        geom_point(aes(color = class)) +
        geom_smooth(
                data = filter(mpg, class == "subcompact"), se = T)

# barplots
# result is the "count" of cuts by row in your data frame!
# this is a new calculated value - like histograms
# these new values are calculated using 'stat' or statistical transformation in R
ggplot(data = diamonds) +
        geom_bar(aes(x = cut))

# if you want you can substitute the geom for the stat operation
ggplot(data = diamonds) +
        stat_count(aes(x = cut))

# building a graph with proportion instead of count
ggplot(data = diamonds) +
        geom_bar(aes(x = cut, y = ..prop.., group = 1))

# building  a graph with stat_summary
# gives you the max, min and median in the graph of cut by depth
# ggplot has over 20+ stat_functions you can use to detail your graphs
ggplot(data = diamonds) +
        stat_summary(aes(x = cut, y = depth),
                     fun.ymin = min,
                     fun.ymax = max,
                     fun.y = median)

# changing colors with barplots
# use the 'fill' command 

# color only shades the outside borders
ggplot(data = diamonds) +
        geom_bar(aes(x = cut, color = cut))

# fill shades the entire bar!
ggplot(data = diamonds) +
        geom_bar(aes(x = cut, fill = cut))

# add fill = clarity option to see stacked bar plot by cut
ggplot(data = diamonds) +
        geom_bar(aes(x = cut, fill = clarity))

# position arguement = "identity"
# places the graphs as the order they "come in" from the dataset
ggplot(data = diamonds, aes(x = cut, fill = clarity)) +
        geom_bar(alpha = 1/5, position = "identity")


ggplot(data = diamonds, aes(x = cut, color = clarity)) +
        geom_bar(fill = NA, position = "identity")

# position arguement = "fill"
# 100% stacked bar chart - easier to compare proportions across groups
ggplot(data = diamonds) +
        geom_bar(aes(x = cut, fill = clarity),
                 position = "fill")
# position arguement = "dodge"
# places individual objects right next to each other - easier to compare values within a plotted group
ggplot(data = diamonds) +
        geom_bar(aes(x = cut, fill = clarity),
                 position = "dodge")

# position arguement = "jitter" a.k.a geom_jitter
# jitter helps prevent overplotting by adding a small amount of random noise to the values
# this helps plot all points so there is no overlap
# small amounts of randomess make data more revealing in the large scale
# geom_jitter implements this function as a geom_function = you will not have to call the position = "jitter"
ggplot(data = mpg) +
        geom_point(aes(x = displ, y = hwy),
                   position = "jitter")

ggplot(data = mpg) +
        geom_jitter(aes(x = displ, y = hwy))



# Coordinate Systems
# default = cartesian coordinate system where x and y position act independently to find the location of each point
# other coordinate systems include: coord_flip(), coord_quickmap(), coord_polar

# coord_flip() example = switches the x and y axes
ggplot(data = mpg, aes(x = class, y = hwy)) +
        geom_boxplot()

#flip axes
ggplot(data = mpg, aes(x = class, y = hwy)) +
        geom_boxplot() +
        coord_flip()

# coord_polar example = uses poloar coordinates

bar = ggplot(data = diamonds) +
        geom_bar(aes(x = cut, fill = cut),
                 show.legend = F, 
                 width = 1) +
        theme(aspect.ratio = 1) +
        labs(x = NULL, y = NULL)
bar

# horitzonal bar chart
bar + coord_flip()

#sliced pie chart = "Coxcomb" chart
bar + coord_polar()


# exercise example:
ggplot(data = mpg, aes(x = cty, y = hwy)) +
        geom_point() +
        geom_abline() +
        coord_fixed()


## updated ggplot foundation example
# you can use this template to build any graph imaginable

# ggplot(data = <DATA>) +
#         <GEOM_FUNCTION>(
#                 mapping = aes(<MAPPINGS>),
#                 stat = <STAT>,
#                 position = <POSITION>
#         ) +
#         <COORDINATE_FUNCTION> +
#         <FACET_FUNCTION>


## grammar of graphics  = you can make any plot based on a combination of:
        # dataset
        # geom function
        # set of mappings
        # a stat operation
        # a position adjustment
        # a coordinate system
        # a faceting scheme





# chapter 3 data transformation -------------------------------------------





## CHAPTER 3: Data transformation in dplyr 
# to explore dplyr we will use the nyc flights dataset

# install needed packages here
install.packages("nycflights13")
library(nycflights13)
library(dplyr)
library(ggplot2)

# tibbles are dataframes that are slightly tweaked to work better in tidyverse
# int = integers
# dbl = double, real numbers
# chr = character or string
# dttm = date-times
# lgl = logical
# fctr = factor; seperates categorical values with fixed possible responses
# date = date

# dplyr basics: five key entry level functions
# filter, arrange, select, mutate, summarize (and group_by)
# these are called "verbs" in dplyr
# all verbs are applied the same way:
        # first arguement is a data frame
        # the subsequent arguements describe what to do with the data
        # the result is a new data frame

# Filter = filter the rows in the data frame

# get all flights on January 1st
# dplyr operations never modify the original data - need to save the data as an object
filter(flights, month == 1, day == 1)

# put the filtered data into variable jan1
jan1 <- filter(flights, month == 1, day ==1)

# print and filter = wrap the entire string in parenthesis
(dec25 <- filter(flights, month == 12, day ==25))

# ALL DPLYR functions for comparisions need to take the '==' operator
# multiple arguements can be passed through to filter = need to use logical operators
# & = "and", | = "or", ! is "not"

# filter values to flights in November or December
# need to specific the second month variable after the "or" operator "|"
filter(flights, month == 11 | month ==12)

# we can also use shorthand %in% for these problems
nov_Dec <- filter(flights, month  %in% c(11, 12))

# using the "not" operator - get flights not in the set defined in filter statement
filter(flights, !(arr_delay > 120 | dep_delay > 120))
filter(flights, arr_delay <= 120 | dep_delay <= 120)

# missing values 
# missing values are coded as NA "not available"
# missing values are contagious and cause full operations to give back NA

# find a value that is missing = is.na()
x = NA
is.na(x)
# [1] TRUE

# filter only includes rows where the condition given is TRUE
# it will exclude both FALSE and NA
# if you want to get back the NA values you will need to ask for them

df <- tibble(x = c(1, NA, 3))
filter(df, is.na(x) | x > 1)
# # A tibble: 2 x 1
# x
# <dbl>
#         1    NA
# 2     3

# Arranging rows
# instead of selecting rows, arrange changes the order of the rows
# order by columns given to the arrange function
arrange(flights, year, month, day)

# use desc() to order in descending order of a variable
arrange(flights, desc(arr_delay))

# missing values are always sorted at the end
df <- tibble(x = c(5,2,NA))
arrange(df, x)

# A tibble: 3 x 1
# x
# <dbl>
#         1     2
# 2     5
# 3    NA

arrange(df, desc(x))
# A tibble: 3 x 1
# x
# <dbl>
#         1     5
# 2     2
# 3    NA



# select columns with dplyr::select
# select allows you to get a subset of data frame columns you want to keep
select(flights, year, month, day)

# select a group of consecutive columns
select(flights, year:day)

# select all column except a group of consecutive columns
select(flights, -(year:day))

# also see some helper functions
# starts_with("abc")
# ends_with("abc")
# contains("abc")
# matches("(.)\\1") = matches on a regular expression
# num_range("x", 1:3) = selects columns 1 through 3

# we can rename column headers with "rename"
rename(flights, tail_num = tailnum)

# example of the "everything" helper
# this move the variables you want to see to the front of the data frame
select(flights, time_hour, air_time, everything())


# mutate = add new variables 
# new variable will be placed at the end of the data frame
# you can do math in mutate as operations of other variables already in the data frame
# you can also refer to the newest columns you just created with mutate

flights.small <- select(flights,
                        year:day,
                        ends_with("delay"),
                        distance, air_time)

# add gain variable to the flights dataset
mutate(flights.small,
       gain = arr_delay - dep_delay,
       speed = distance / air_time * 60)

# refer to a variable made from a previous mutate statment
mutate(flights.small, 
       gain = arr_delay - dep_delay,
       speed = distance / air_time * 60,
       hours = air_time / 60,
       gain_ph = gain / hours)

# if you only want to keep the new variables you created = TRANSMUTE
transmute(flights,
          gain = arr_delay - dep_delay,
          hours = air_time / 60,
          gain_ph = gain / hours)

# useful creation functions = all these can be used in mutate!!
# R provides many functions you can use to create new variables
# the inputs to the functions must be VECTORIZED: it must take vector as input, and return a vector with the same number of items
        # arithmetic functions: +, -, *, /, ^
        # modulat arithmetic = integer division "%/%"  and remainder "%%"
        # logs = logarithms transformation to different data
        # offsets = lead(), and lag() allow you to refer to a leading or lagging values
(x <-  1:10)
lag(x)
lead(x)
        # cumlulative / rolling aggregates = cumsum, cumprod, cummin, cumax, cummean = operations over a rolling window
x        
cumsum(x)
cummean(x)
        # logical comparisons = <, <=, >, >=, !=
        # ranking = various ranking functions can be passed through mutate = try min_rank)
y <- c(1,2,2,NA,3,4)
min_rank(y)
min_rank(desc(y))
row_number(y)
dense_rank(y)
percent_rank(y)
cume_dist(y)



# grouped summaries with summmarize()
# summarize collapses data into a single row

summarize(flights, delay = mean(dep_delay, na.rm = T))
# A tibble: 1 x 1
# delay
# <dbl>
#         1 12.63907

# need to combine summarize with group_by to make it useful...summarise will group all down to one row unless told which groups to "group by"
# group_by changes the unit of analysis from the whole dataset to selected groups
# any dplyr verbs applied after a group_by statement will pass that verb through the group_by defined groups!!

# intial group_by statement
by_day <- group_by(flights, year, month, day)

# summarise statement passed through to the grouped data frame
summarize(by_day, delay = mean(dep_delay, na.rm = T))

# A tibble: 365 x 4
# Groups:   year, month [?]
# year month   day     delay
# <int> <int> <int>     <dbl>
#         1  2013     1     1 11.548926
# 2  2013     1     2 13.858824
# 3  2013     1     3 10.987832
# 4  2013     1     4  8.951595
# 5  2013     1     5  5.732218
# 6  2013     1     6  7.148014
# 7  2013     1     7  5.417204
# 8  2013     1     8  2.553073
# 9  2013     1     9  2.276477
# 10  2013     1    10  2.844995
# # ... with 355 more rows



# piping dplyr statements together

# code before piping
# this takes three seperate steps to get all the way to the end
        # group by destination
        # summarize to compute the needed metrics
        # filter to remove certain data points
by_dest <- group_by(flights, dest)

delay <- summarize(by_dest,
                   count = n(),
                   dist = mean(distance, na.rm = T),
                   delay = mean(arr_delay, na.rm = T))

delay <- filter(delay, count >20, dest != "HNL")

ggplot(data = delay, aes(x = dist, y = delay)) +
        geom_point(aes(size = count), alpha = 1/3) +
        geom_smooth(se = T)

# same code as above but with piping " %>% " operator
# this focuses your code on the transformations, not what's being transformed = easier to read
# use piping operatior as a "then" statement
delays <- flights %>% 
        group_by(dest) %>% 
        summarize(
                count = n(),
                dist = mean(distance, na.rm = T),
                delay = mean(arr_delay, na.rm = T
        )) %>% 
        filter(count > 20, dest != "HNL") %>% 
        ggplot(data =  ., aes(x = dist, y = delay)) +
        geom_point(aes(size = count), alpha = 1/3) +
        geom_smooth(se = T)

# side note: make sure to subset out NAs from the data
# dplyr verbs will react to this outbreak and give you back only NAs for all verbs
# na.rm = T removes all NAs from the verb's action

flights %>% 
        group_by(year, month, day) %>% 
        summarize(mean = mean(dep_delay, na.rm = T))

not_cancelled <- flights %>% 
        filter(!is.na(dep_delay), !is.na(arr_delay))

not_cancelled %>% 
        group_by(year, month, day) %>% 
        summarize(mean = mean(dep_delay))
# A tibble: 365 x 4
# Groups:   year, month [?]
# year month   day      mean
# <int> <int> <int>     <dbl>
#         1  2013     1     1 11.435620
# 2  2013     1     2 13.677802
# 3  2013     1     3 10.907778
# 4  2013     1     4  8.965859
# 5  2013     1     5  5.732218
# 6  2013     1     6  7.145959
# 7  2013     1     7  5.417204
# 8  2013     1     8  2.558296
# 9  2013     1     9  2.301232
# 10  2013     1    10  2.844995
# ... with 355 more rows


# counts
# whenever you do an aggregation - include a count (n()) or a count of nonmissing values (sum(!is.na(x)))
# this way you can check to make sure you are not drawing conclusions on small sample size
# example below:
delays <- not_cancelled %>% 
        group_by(tailnum) %>% 
        summarize(
                delay = mean(arr_delay)
        )

# plot shows plane delays over 5 hours...is this true? Maybe there are a couple of large outliers weighing out the set
ggplot(data = delays, aes(x = delay)) +
        geom_freqpoly(binwidth = 10)

# more inspection - lets get the count of each to plot delay time by number of times that delay happened
delays <- not_cancelled %>% 
        group_by(tailnum) %>% 
        summarize(
                delay = mean(arr_delay, na.rm = T),
                n = n()
        )

# plot flights vs. delay
# there is MORE VARIATION when there ARE FEWER FLIGHTS!!!!
# VARIATION WILL DECREASE AS YOU INCREASE SAMPLE SIZE
ggplot(data = delays, aes(x = n, y = delay)) +
        geom_point(alpha = 1/10)

# putting dplyr and ggplot together
# take out flights with count less than 25 to smooth remove some of the low sample variance
delays %>% 
        filter(n > 25) %>% 
        ggplot(aes(x = n, y = delay)) +
        geom_point(alpha = 1/10)




# useful summary functions to use within the summarize verb!!
# just using means, counts and sum can get you a long way - but there are many more summarize functions available!!
library(dplyr)

# measures of central tendancy or location
not_cancelled %>% 
        group_by(year, month, day) %>% 
        summarize(
                avg_delay1 = mean(arr_delay),
                avg_delay2 = median(arr_delay)
        )

# # A tibble: 365 x 5
# # Groups:   year, month [?]
# year month   day avg_delay1 avg_delay2
# <int> <int> <int>      <dbl>      <dbl>
#         1  2013     1     1 12.6510229          3
# 2  2013     1     2 12.6928879          4
# 3  2013     1     3  5.7333333          1
# 4  2013     1     4 -1.9328194         -8
# 5  2013     1     5 -1.5258020         -7
# 6  2013     1     6  4.2364294         -1
# 7  2013     1     7 -4.9473118        -10
# 8  2013     1     8 -3.2275785         -7
# 9  2013     1     9 -0.2642777         -6
# 10  2013     1    10 -5.8988159        -11
# # ... with 355 more rows


# measures of spread
not_cancelled %>% 
        group_by(year, month, day) %>% 
        summarize(
                distance_sd = sd(distance)) %>% 
        arrange(desc(distance_sd))

# # A tibble: 365 x 4
# # Groups:   year, month [12]
# year month   day distance_sd
# <int> <int> <int>       <dbl>
#         1  2013     5    23    780.3787
# 2  2013     8    31    774.8431
# 3  2013     9     7    772.4758
# 4  2013     9     1    772.3673
# 5  2013     7    10    771.3746
# 6  2013     9    21    770.5352
# 7  2013     9    14    770.4390
# 8  2013     6    27    768.9369
# 9  2013     7    22    767.7016
# 10  2013     6    28    767.5768
# # ... with 355 more rows


# measures of rank
not_cancelled %>% 
        group_by(year, month, day) %>% 
        summarize(
                first = max(dep_time),
                last = min(dep_time)
        )

# # A tibble: 365 x 5
# # Groups:   year, month [?]
# year month   day first  last
# <int> <int> <int> <dbl> <dbl>
#         1  2013     1     1  2356   517
# 2  2013     1     2  2354    42
# 3  2013     1     3  2349    32
# 4  2013     1     4  2358    25
# 5  2013     1     5  2357    14
# 6  2013     1     6  2355    16
# 7  2013     1     7  2359    49
# 8  2013     1     8  2351   454
# 9  2013     1     9  2252     2
# 10  2013     1    10  2320     3
# # ... with 355 more rows


# measures of position
not_cancelled %>% 
        group_by(year, month, day) %>% 
        summarize(
                first_dep = first(dep_time), 
                last_dep = last(dep_time)
        )

# # A tibble: 365 x 5
# # Groups:   year, month [?]
# year month   day first_dep last_dep
# <int> <int> <int>     <int>    <int>
#         1  2013     1     1       517     2356
# 2  2013     1     2        42     2354
# 3  2013     1     3        32     2349
# 4  2013     1     4        25     2358
# 5  2013     1     5        14     2357
# 6  2013     1     6        16     2355
# 7  2013     1     7        49     2359
# 8  2013     1     8       454     2351
# 9  2013     1     9         2     2252
# 10  2013     1    10         3     2320
# # ... with 355 more rows


# filter on ranks based on mutate variable
not_cancelled %>% 
        group_by(year, month, day) %>% 
        mutate(r = min_rank(desc(dep_time))) %>% 
        filter(r %in% range(r))
# # A tibble: 770 x 20
# # Groups:   year, month, day [365]
# year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time arr_delay carrier flight tailnum origin  dest air_time distance  hour minute           time_hour     r
# <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>     <dbl>   <chr>  <int>   <chr>  <chr> <chr>    <dbl>    <dbl> <dbl>  <dbl>              <dttm> <int>
#         1  2013     1     1      517            515         2      830            819        11      UA   1545  N14228    EWR   IAH      227     1400     5     15 2013-01-01 05:00:00   831
# 2  2013     1     1     2356           2359        -3      425            437       -12      B6    727  N588JB    JFK   BQN      186     1576    23     59 2013-01-01 23:00:00     1
# 3  2013     1     2       42           2359        43      518            442        36      B6    707  N580JB    JFK   SJU      189     1598    23     59 2013-01-02 23:00:00   928
# 4  2013     1     2     2354           2359        -5      413            437       -24      B6    727  N789JB    JFK   BQN      180     1576    23     59 2013-01-02 23:00:00     1
# 5  2013     1     3       32           2359        33      504            442        22      B6    707  N763JB    JFK   SJU      193     1598    23     59 2013-01-03 23:00:00   900
# 6  2013     1     3     2349           2359       -10      434            445       -11      B6    739  N729JB    JFK   PSE      199     1617    23     59 2013-01-03 23:00:00     1
# 7  2013     1     4       25           2359        26      505            442        23      B6    707  N554JB    JFK   SJU      194     1598    23     59 2013-01-04 23:00:00   908
# 8  2013     1     4     2358           2359        -1      429            437        -8      B6    727  N599JB    JFK   BQN      189     1576    23     59 2013-01-04 23:00:00     1
# 9  2013     1     4     2358           2359        -1      436            445        -9      B6    739  N821JB    JFK   PSE      199     1617    23     59 2013-01-04 23:00:00     1
# 10  2013     1     5       14           2359        15      503            445        18      B6    739  N592JB    JFK   PSE      201     1617    23     59 2013-01-05 23:00:00   717
# # ... with 760 more rows

# counts
not_cancelled %>% 
        group_by(dest) %>% 
        summarize(carriers = n_distinct(carrier)) %>% 
        arrange(desc(carriers))
# # A tibble: 104 x 2
# dest carriers
# <chr>    <int>
#         1   ATL        7
# 2   BOS        7
# 3   CLT        7
# 4   ORD        7
# 5   TPA        7
# 6   AUS        6
# 7   DCA        6
# 8   DTW        6
# 9   IAD        6
# 10   MSP        6
# # ... with 94 more rows



# simple count in dplyr
not_cancelled %>% count(dest)
# # A tibble: 104 x 2
# dest     n
# <chr> <int>
#         1   ABQ   254
# 2   ACK   264
# 3   ALB   418
# 4   ANC     8
# 5   ATL 16837
# 6   AUS  2411
# 7   AVL   261
# 8   BDL   412
# 9   BGR   358
# 10   BHM   269
# # ... with 94 more rows


# simple count with weight option - sums distance by tailnum
not_cancelled %>% count(tailnum, wt = distance)
# # A tibble: 4,037 x 2
# tailnum      n
# <chr>  <dbl>
#         1  D942DN   3418
# 2  N0EGMQ 239143
# 3  N10156 109664
# 4  N102UW  25722
# 5  N103US  24619
# 6  N104UW  24616
# 7  N10575 139903
# 8  N105UW  23618
# 9  N107US  21677
# 10  N108UW  32070
# # ... with 4,027 more rows


# counts and proportions of logical values
# sum converts TRUE values to 1 and FALSE to 0 = allows summing by the TRUE variables!!!
not_cancelled %>% 
        group_by(year, month, day) %>% 
        summarize(n_early = sum(dep_time < 500))
# # A tibble: 365 x 4
# # Groups:   year, month [?]
# year month   day n_early
# <int> <int> <int>   <int>
#         1  2013     1     1       0
# 2  2013     1     2       3
# 3  2013     1     3       4
# 4  2013     1     4       3
# 5  2013     1     5       3
# 6  2013     1     6       2
# 7  2013     1     7       2
# 8  2013     1     8       1
# 9  2013     1     9       3
# 10  2013     1    10       3
# # ... with 355 more rows

# mean with logical statement
not_cancelled %>% 
        group_by(year, month, day) %>% 
        summarize(hour_perc = mean(arr_delay > 60))
# # A tibble: 365 x 4
# # Groups:   year, month [?]
# year month   day  hour_perc
# <int> <int> <int>      <dbl>
#         1  2013     1     1 0.07220217
# 2  2013     1     2 0.08512931
# 3  2013     1     3 0.05666667
# 4  2013     1     4 0.03964758
# 5  2013     1     5 0.03486750
# 6  2013     1     6 0.04704463
# 7  2013     1     7 0.03333333
# 8  2013     1     8 0.02130045
# 9  2013     1     9 0.02015677
# 10  2013     1    10 0.01829925
# # ... with 355 more rows


# grouping by multiple variables
# when you group by multiple variables, each summary peels off one level of the grouping
# this makes it easy to progressively roll up the dataset

daily <- group_by(flights.small, year, month, day)
(per_day <- summarize(daily, flights = n()))

# # A tibble: 365 x 4
# # Groups:   year, month [?]
# year month   day flights
# <int> <int> <int>   <int>
#         1  2013     1     1     842
# 2  2013     1     2     943
# 3  2013     1     3     914
# 4  2013     1     4     915
# 5  2013     1     5     720
# 6  2013     1     6     832
# 7  2013     1     7     933
# 8  2013     1     8     899
# 9  2013     1     9     902
# 10  2013     1    10     932
# # ... with 355 more rows

(per_month <- summarize(per_day, flights = sum(flights)))
# # A tibble: 12 x 3
# # Groups:   year [?]
# year month flights
# <int> <int>   <int>
#         1  2013     1   27004
# 2  2013     2   24951
# 3  2013     3   28834
# 4  2013     4   28330
# 5  2013     5   28796
# 6  2013     6   28243
# 7  2013     7   29425
# 8  2013     8   29327
# 9  2013     9   27574
# 10  2013    10   28889
# 11  2013    11   27268
# 12  2013    12   28135

(per_year <- summarize(per_month, flights = sum(flights)))
# # A tibble: 1 x 2
# year flights
# <int>   <int>
#         1  2013  336776


# ungrouping = if you need to remove grouping and return to operations on ungrouped data, use ungroup()
daily %>% 
        ungroup() %>% 
        summarize(flights = n())
# # A tibble: 1 x 1
# flights
# <int>
#         1  336776


# grouped mutates
# grouping is most useful in conjuntion with summarize(), but you can also do convenient operations with mutate() and filter()

# find the worst members of each group
flights %>% 
        group_by(year, month, day) %>% 
        filter(rank(desc(arr_delay)) < 10)

# # A tibble: 3,306 x 19
# # Groups:   year, month, day [365]
# year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time arr_delay carrier flight tailnum origin  dest air_time distance  hour minute           time_hour
# <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>     <dbl>   <chr>  <int>   <chr>  <chr> <chr>    <dbl>    <dbl> <dbl>  <dbl>              <dttm>
#         1  2013     1     1      848           1835       853     1001           1950       851      MQ   3944  N942MQ    JFK   BWI       41      184    18     35 2013-01-01 18:00:00
# 2  2013     1     1     1815           1325       290     2120           1542       338      EV   4417  N17185    EWR   OMA      213     1134    13     25 2013-01-01 13:00:00
# 3  2013     1     1     1842           1422       260     1958           1535       263      EV   4633  N18120    EWR   BTV       46      266    14     22 2013-01-01 14:00:00
# 4  2013     1     1     1942           1705       157     2124           1830       174      MQ   4410  N835MQ    JFK   DCA       60      213    17      5 2013-01-01 17:00:00
# 5  2013     1     1     2006           1630       216     2230           1848       222      EV   4644  N14972    EWR   SAV      121      708    16     30 2013-01-01 16:00:00
# 6  2013     1     1     2115           1700       255     2330           1920       250      9E   3347  N924XJ    JFK   CVG      115      589    17      0 2013-01-01 17:00:00
# 7  2013     1     1     2205           1720       285       46           2040       246      AA   1999  N5DNAA    EWR   MIA      146     1085    17     20 2013-01-01 17:00:00
# 8  2013     1     1     2312           2000       192       21           2110       191      EV   4312  N13958    EWR   DCA       44      199    20      0 2013-01-01 20:00:00
# 9  2013     1     1     2343           1724       379      314           1938       456      EV   4321  N21197    EWR   MCI      222     1092    17     24 2013-01-01 17:00:00
# 10  2013     1     2     1244            900       224     1431           1104       207      EV   4412  N13958    EWR   MYR       94      550     9      0 2013-01-02 09:00:00
# # ... with 3,296 more rows


# find all groups bigger than a threshold
popular_dest <- flights %>% 
        group_by(dest) %>% 
        filter(n() > 365)
popular_dest

# # A tibble: 332,577 x 19
# # Groups:   dest [77]
# year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time arr_delay carrier flight tailnum origin  dest air_time distance  hour minute           time_hour
# <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>     <dbl>   <chr>  <int>   <chr>  <chr> <chr>    <dbl>    <dbl> <dbl>  <dbl>              <dttm>
#         1  2013     1     1      517            515         2      830            819        11      UA   1545  N14228    EWR   IAH      227     1400     5     15 2013-01-01 05:00:00
# 2  2013     1     1      533            529         4      850            830        20      UA   1714  N24211    LGA   IAH      227     1416     5     29 2013-01-01 05:00:00
# 3  2013     1     1      542            540         2      923            850        33      AA   1141  N619AA    JFK   MIA      160     1089     5     40 2013-01-01 05:00:00
# 4  2013     1     1      544            545        -1     1004           1022       -18      B6    725  N804JB    JFK   BQN      183     1576     5     45 2013-01-01 05:00:00
# 5  2013     1     1      554            600        -6      812            837       -25      DL    461  N668DN    LGA   ATL      116      762     6      0 2013-01-01 06:00:00
# 6  2013     1     1      554            558        -4      740            728        12      UA   1696  N39463    EWR   ORD      150      719     5     58 2013-01-01 05:00:00
# 7  2013     1     1      555            600        -5      913            854        19      B6    507  N516JB    EWR   FLL      158     1065     6      0 2013-01-01 06:00:00
# 8  2013     1     1      557            600        -3      709            723       -14      EV   5708  N829AS    LGA   IAD       53      229     6      0 2013-01-01 06:00:00
# 9  2013     1     1      557            600        -3      838            846        -8      B6     79  N593JB    JFK   MCO      140      944     6      0 2013-01-01 06:00:00
# 10  2013     1     1      558            600        -2      753            745         8      AA    301  N3ALAA    LGA   ORD      138      733     6      0 2013-01-01 06:00:00
# # ... with 332,567 more rows


# standardize to compute per group metrics
popular_dest %>% 
        filter(arr_delay > 0) %>% 
        mutate(prop_delay = arr_delay / sum(arr_delay)) %>% 
        select(year:day, dest, arr_delay, prop_delay)
# # A tibble: 131,106 x 6
# # Groups:   dest [77]
# year month   day  dest arr_delay   prop_delay
# <int> <int> <int> <chr>     <dbl>        <dbl>
#         1  2013     1     1   IAH        11 1.106740e-04
# 2  2013     1     1   IAH        20 2.012255e-04
# 3  2013     1     1   MIA        33 2.350026e-04
# 4  2013     1     1   ORD        12 4.239594e-05
# 5  2013     1     1   FLL        19 9.377853e-05
# 6  2013     1     1   ORD         8 2.826396e-05
# 7  2013     1     1   LAX         7 3.444441e-05
# 8  2013     1     1   DFW        31 2.817951e-04
# 9  2013     1     1   ATL        12 3.996017e-05
# 10  2013     1     1   DTW        16 1.157257e-04
# # ... with 131,096 more rows



# chapter 4 workflow ------------------------------------------------------



## Chapter 4: Workflow Scripts
# control - shift - N gives you a new text editor


# chapter 5 EDA -----------------------------------------------------------



## Chapter 5: Exploratory Data Analysis


# visualize and transform data to explore in a systematic way = exploratory data analysis
        # generate questions about your data
        # search for answers by visualizing, transforming, and modeling your data
        # use what you learn to refine your questions or generate new questions

# EDA is a state of mind - feel free to investigate any aspect of your data - some will work some will be dead ends
# idea is to hone in on a few particularly useful ideas to communicate out to the group
# you also need to investigate the quality of your data
        # tools of EDA = visualization, transformation, modeling

install.packages("tidyverse")
library(tidyverse)
library(ggplot2)
library(plyr)
library(dplyr)


# there are no routine statistical questions, only questionable statistical routines
# far better to approximate an exact answer to the right questionm which is often vague, than an exact answer to the wrong question, which can always be made precise

# goal: develop an understanding of the data = use questions and tools to guide your exploration
# when you ask a question - it forces you to transform your data into the format you need
# key is to generate a large quantitiy of questions to investigate

# two types of questions to always check out:
        # what type of variation occurs within my variables?
        # what type of covariation occurs between by variables?

# variable - quantity, quality or property that you can measure
# value is the state of a variable when you measure it
# observation - set of measurements made under similiar conditions = data point
# tabular data - set of values, each associated with a variable or observation - needs to be tidy


## Variation
# variation is the tendency of the values of a variable to change from measurement to measurement "spread"
# each measurement contains some error that differs on a case by case basis
# every variable has its own pattern of variation - these can reveal interesting information
# the best way to understand this is to visualize the distribution of the variable's values

## Visualizing Distributions
        # depends on if your data is continous or categorical
                # categorical = data can only take a finite set of values
                # continous = data can take on an infinite set of ordered values

# categorical visualization: bar chart = shows count by category
ggplot(data = diamonds) +
        geom_bar(aes(x = cut))


# calculate manually:

diamonds %>% count(cut)

# # A tibble: 5 x 2
# cut     n
# <ord> <int>
#         1      Fair  1610
# 2      Good  4906
# 3 Very Good 12082
# 4   Premium 13791
# 5     Ideal 21551

# continous variable: histogram = shows count of variable by "bins" of possible values
ggplot(data = diamonds) +
        geom_histogram(aes(x = carat), binwidth = 0.5)

# compute by hand...
diamonds %>% count(cut_width(carat, 0.5))

# A tibble: 11 x 2
# `cut_width(carat, 0.5)`     n
# <fctr> <int>
#         1            [-0.25,0.25]   785
# 2             (0.25,0.75] 29498
# 3             (0.75,1.25] 15977
# 4             (1.25,1.75]  5313
# 5             (1.75,2.25]  2002
# 6             (2.25,2.75]   322
# 7             (2.75,3.25]    32
# 8             (3.25,3.75]     5
# 9             (3.75,4.25]     4
# 10             (4.25,4.75]     1
# 11             (4.75,5.25]     1


# historgram divides x axis into equally spaced bins and uses the height of each bar to display the number of observations that fall into that bin
# you can set the values of the "bin" with the binwidth function
# try different binwidths - might reveal some interesting points about your data
small <- diamonds %>% 
        filter(carat < 1)
ggplot(data = small, aes(x = carat))+
        geom_histogram(binwidth = 0.1)



# overlay multiple histograms in the same plot = USE GEOM_FREQPLOTY
# uses lines instead so we can compare the continous distribution for multiple variables
ggplot(data = small, aes(x = carat, color = cut)) +
        geom_freqpoly(binwidth = 0.1)

## Typical Values
# both bar charts and histograms, tall bars show common values of a variable, short bars so more rare occurances 
# places without bars show where we do not have any data!
# ask: which values are most common? why? are there unusual patterns?

# example:
        # why are there more diamonds at whole carats and common fractions of carats?
        # why are diamonds more common slightly to the right of each peak?
        # why are there no diamonds bigger than 3 carats?
# in general clusters of similiar repsonses suggest subgroups exist in your data
ggplot(data = small, aes(x = carat)) +
        geom_histogram(binwidth = .01)


# how are observations within each cluster similiar?
# how are the observations in seperate clusters different?
# how can we explain this?
# why might the appearance of these clusters be misleading?

# example: eruptions histogram
ggplot(data = faithful, aes(x = eruptions)) +
        geom_histogram(binwidth = .25)

# data shows two main clusters centered around 2 and 4,5 minutes...nothing else inbetween...why?
# many of these questions will lead you to explore a relationship between variables


## Unusual Values: Outliers
# outliers are observations that are unusual, data points that don't seem to fit the problem
# sometimes outliers are cloudy on histograms due to binwidth
ggplot(diamonds) +
        geom_histogram(aes(x = y), binwidth = .5)

# there are so many observations it is hard to see any outliers - replot data
# all the sudden more outliers appear!
# coord cartesian lets you zoom in on the very small values of the dataset
# you can also use coord_cartesian with an xlim arguement!
ggplot(diamonds) +
        geom_histogram(aes(x = y), binwidth = .5) +
        coord_cartesian(ylim = c(0,50))

# based on the plot above - we can investigate the unusal points (0,30,60)
unusual <- diamonds %>% 
        filter(y <3 | y > 20) %>% 
        arrange(y)

# # A tibble: 9 x 10
# carat       cut color clarity depth table price     x     y     z
# <dbl>     <ord> <ord>   <ord> <dbl> <dbl> <int> <dbl> <dbl> <dbl>
#         1  1.00 Very Good     H     VS2  63.3    53  5139  0.00   0.0  0.00
# 2  1.14      Fair     G     VS1  57.5    67  6381  0.00   0.0  0.00
# 3  1.56     Ideal     G     VS2  62.2    54 12800  0.00   0.0  0.00
# 4  1.20   Premium     D    VVS1  62.1    59 15686  0.00   0.0  0.00
# 5  2.25   Premium     H     SI2  62.8    59 18034  0.00   0.0  0.00
# 6  0.71      Good     F     SI2  64.1    60  2130  0.00   0.0  0.00
# 7  0.71      Good     F     SI2  64.1    60  2130  0.00   0.0  0.00
# 8  0.51     Ideal     E     VS1  61.8    55  2075  5.15  31.8  5.12
# 9  2.00   Premium     H     SI2  58.9    57 12210  8.09  58.9  8.06


# you may want to repeat your analysis with the outliers removed
# but you need to explain them and justify thier removal


## Missing Values:
# if you have bad data you can drop the entire row from the dataset
# just becuase one measurement is invalid, doesn't mean they all are!! be careful!!
diamonds2 <- diamonds %>% 
        filter(between(y, 3,20))
# instead you can replace unusal values with missing values: use mutate command
diamonds2 <- diamonds %>% 
        mutate(y = ifelse(y <3 | y > 20, NA, y))

#ifelse = (test, val if true, val if false)
# ggplot = values never go silently missing = it will drop a warning
ggplot(data = diamonds2, aes(x = x, y = y)) +
        geom_point(na.rm = F)

# Warning message:
#         Removed 9 rows containing missing values (geom_point). 

# na.rm = T will remove the missing values in your plot...BE CAREFUL!!!!


# example: nyc flights: comparing observed values to missing values
nycflights13::flights %>% 
        mutate(
                cancelled = is.na(dep_time),
                sched_hour = sched_dep_time %/% 100,
                sched_min = sched_dep_time %/% 100,
                sched_dep_time = sched_hour + sched_min / 60
        ) %>% 
        ggplot(aes(sched_dep_time)) +
        geom_freqpoly(
                aes(color = cancelled),
                binwidth = 1/4
        )


## Covariation

# if variation describes behavior within a variable 
# covarition is the the behavior between variables
# two or more variables can vary togther in the same way
# to spot covarition = visual the relationship between two or more variables

## categorical and continous variable

# freqploty doesn't account for the size of each variable
ggplot(diamonds, aes(x = price)) +
        geom_freqpoly(aes(color = cut), binwidth = 500)

# even with bar graphs you cannot see between the difference is size for each variable
ggplot(diamonds)+
        geom_bar(aes(x = cut))

# to make these comparisions easier - we need to swap out what the measure is on the y axis
ggplot(data = diamonds, aes(x = price, y = ..density..)) +
        geom_freqpoly(aes(color = cut), binwidth = 500)

# another way to interpret this data is to use a boxplot
# box = 25th to 75th percentile = IQR
# median is the middle of the box
# information gives you the spread of each variable = distribution = is it skewed?
# points on the boxplot show the outliers of the data = they fall more than the IQR away form the median
# the whisker points to the furthest non-outlier point
ggplot(diamonds, aes(x = cut, y = price)) +
        geom_boxplot()

# we can reorder the variables to gain more intuitive information
ggplot(data = mpg, aes(x = class, y = hwy)) +
        geom_boxplot()

ggplot(data = mpg)+
        geom_boxplot(aes(x = reorder(class, hwy, FUN = median), y = hwy))

# you can coord flip the boxplot to see the long x axis values
ggplot(data = mpg)+
        geom_boxplot(aes(x = reorder(class, hwy, FUN = median), y = hwy)) +
        coord_flip()

## two categorical variables
# need to count the number of observations for each combination

#built in geom_count
# size of the circle shows how many times an observation is shown
# covarition will show as a strong positive relationship between two values
ggplot(diamonds)+
        geom_count(aes(x = cut, y = color))

# using dplyr count
diamonds %>% count(color, cut)

# # A tibble: 35 x 3
# color       cut     n
# <ord>     <ord> <int>
#         1     D      Fair   163
# 2     D      Good   662
# 3     D Very Good  1513
# 4     D   Premium  1603
# 5     D     Ideal  2834
# 6     E      Fair   224
# 7     E      Good   933
# 8     E Very Good  2400
# 9     E   Premium  2337
# 10     E     Ideal  3903
# # ... with 25 more rows

# we can visualize these counts with the geom_tile and fill
# deeper colors will show less counts of the categorical variables against each other
diamonds %>% 
        count(color, cut) %>% 
        ggplot(aes(x = color, y = cut)) +
        geom_tile(aes(fill = n))

## two continous variables
# scatterplot = you can see covariaiton pattern between the points

# relationship between carat and size 
ggplot(diamonds)+
        geom_point(aes(x = carat, y = price))

# scatterplots get tricky when there is a lot of data
# overplotting
# fix with alpha
ggplot(diamonds) +
        geom_point(aes(x = carat, y = price), alpha = 1/100)

# another way to get around this is to "bin" your categorical variable so it acts as continous
ggplot(small) +
        geom_bin2d(aes(x = carat, y = price))

# bin your variable using the cut_wdith arguement
ggplot(small, aes(x = carat, y = price)) +
        geom_boxplot(aes(group = cut_width(carat, 0.1)))

# you can show the number of points in each box using the varwidth option
# show the same amount of points for each box using the cut_number option = 20 obs per box
ggplot(small, aes(x = carat, y = price))+
        geom_boxplot(aes(group = cut_number(carat, 20)))

## patterns and models
# patterns provide clues about the relationship of your data
# does this occur by chance?
# how can we describe this relationship?
# how strong is this relationship?
# what other variables might effect this relationship?
# does this relationship change if you look at other subgroups?

# old faithful
ggplot(data = faithful)+
        geom_point(aes(x = eruptions, y = waiting))
# patterns reveal covarition!
# covariaiton is a pattern that reduces uncertainty
# if variables covary together, you can use one to model your results

# models are tools for extracting patterns from data
# you can use models to remove the covariation variables and then re-explore without one of those variables

install.packages("modelr")
library(modelr)

mod <- lm(log(price) ~ log(carat), data = diamonds)

diamonds2 <-  diamonds %>% 
        add_residuals(mod) %>% 
        mutate(resid = exp(resid))

ggplot(diamonds2)+
        geom_point(aes(x = carat, y = resid))

# carat causes lots of residual error
# lets remove carat and see the results of the diamond dataset
ggplot(diamonds2) +
        geom_boxplot(aes(x = cut, y = resid))

## RELATIVE TO THIER SIZE (CARAT) higher quality diamonds are more expensive!!!






## PART 2: WRANGLE


# Chapter 7 Tibbles -------------------------------------------------------



## Chapter 7: Tibbles

# tibbles are dataframes, but they tweak some older behaviors to make like more easier
# it is difficult to change base R - so upgrades are made with packages
# tibble packages are "opinionated" data frames that make working in the tidyverse easier

install.packages("tibble")
library(tibble)

# almost all functions in the dplyr universe will produce tibbles
# tibbles are one of the unifying features of tidyverse
# you can coherce a data frame to a tibble if you want to
dplyr::as_tibble(iris)

# # A tibble: 150 x 5
# Sepal.Length Sepal.Width Petal.Length Petal.Width Species
# <dbl>       <dbl>        <dbl>       <dbl>  <fctr>
#         1          5.1         3.5          1.4         0.2  setosa
# 2          4.9         3.0          1.4         0.2  setosa
# 3          4.7         3.2          1.3         0.2  setosa
# 4          4.6         3.1          1.5         0.2  setosa
# 5          5.0         3.6          1.4         0.2  setosa
# 6          5.4         3.9          1.7         0.4  setosa
# 7          4.6         3.4          1.4         0.3  setosa
# 8          5.0         3.4          1.5         0.2  setosa
# 9          4.4         2.9          1.4         0.2  setosa
# 10          4.9         3.1          1.5         0.1  setosa
# # ... with 140 more rows


# you can create a new tibble from individual vectors
dplyr::tibble(
        x = 1:5,
        y = 1, 
        z = x^2 + y
)


# # A tibble: 5 x 3
# x     y     z
# <int> <dbl> <dbl>
#         1     1     1     2
# 2     2     1     5
# 3     3     1    10
# 4     4     1    17
# 5     5     1    26


## note that tibble does much less than data.frame:
        # it never changes the type of inputs
        # it never changes the type of the variables or names
        # it never creates row names

# it's possible for tibble to have column names that are not valid R variable names
# to call these variables you need to call them with ``
(tb <- dplyr::tibble(
        `:)` = "smile",
        ` ` = "space",
        `2000` = "number"
))

# # A tibble: 1 x 3
# `:)`   ` ` `2000`
# <chr> <chr>  <chr>
#         1 smile space number

# backticks will also work in other packages like ggplot2, dplyr, tidyr

# another way to write a tibble is to call tribble: TRANSPOSED TIBBLE
# tribble is customized for data entry in code: column headings are defined by formulas, and entries are seperated by commas
# this makes it possible to lay out small amounts of data into an easy to read form
dplyr::tribble(
        ~x, ~y, ~z,
        "a", 2, 3.6,
        "B", 1,  8.5
)

# # A tibble: 2 x 3
# x     y     z
# <chr> <dbl> <dbl>
#         1     a     2   3.6
# 2     B     1   8.5


# tibbles vs. data.frame
# there are two main differences in the usage of a tibble versus a data.frame: PRINTING and SUBSETTING

## PRINTING
# tibbles have a refined print method that shows only the first 10 rows, and all columns fit on the screen
# this makes it easier to work with large data
# in addition to the name, each column also reports its type (borrowed from str())
dplyr::tibble(
        a = lubridate::now() + runif(1e3)*86400,
        b = lubridate::today() + runif(1e3)*30,
        c = 1:1e3,
        d = runif(1e3),
        e = sample(letters, 1e3, replace = T)
)

# # A tibble: 1,000 x 5
# a          b     c           d     e
# <dttm>     <date> <int>       <dbl> <chr>
#         1 2018-01-28 20:39:43 2018-02-12     1 0.730150222     i
# 2 2018-01-29 03:34:40 2018-02-15     2 0.512892896     i
# 3 2018-01-28 20:38:52 2018-01-29     3 0.248496525     p
# 4 2018-01-28 23:45:23 2018-02-14     4 0.004745992     r
# 5 2018-01-29 19:18:47 2018-02-17     5 0.552203702     u
# 6 2018-01-28 20:43:28 2018-02-01     6 0.867789885     a
# 7 2018-01-29 09:19:30 2018-02-19     7 0.225289842     t
# 8 2018-01-28 20:55:24 2018-02-25     8 0.682811185     a
# 9 2018-01-29 06:11:33 2018-02-26     9 0.393309354     u
# 10 2018-01-29 04:45:37 2018-02-04    10 0.268548397     g
# # ... with 990 more rows

library(dplyr)

# tibbles are designed so you do not overwhelm your console with printing large dataframes
# if you need to print the whole thing; try the print option
nycflights13::flights %>% 
        print(n = 10, width = Inf)

# # A tibble: 336,776 x 19
# year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time arr_delay carrier flight tailnum origin  dest air_time distance  hour minute           time_hour
# <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>     <dbl>   <chr>  <int>   <chr>  <chr> <chr>    <dbl>    <dbl> <dbl>  <dbl>              <dttm>
#         1  2013     1     1      517            515         2      830            819        11      UA   1545  N14228    EWR   IAH      227     1400     5     15 2013-01-01 05:00:00
# 2  2013     1     1      533            529         4      850            830        20      UA   1714  N24211    LGA   IAH      227     1416     5     29 2013-01-01 05:00:00
# 3  2013     1     1      542            540         2      923            850        33      AA   1141  N619AA    JFK   MIA      160     1089     5     40 2013-01-01 05:00:00
# 4  2013     1     1      544            545        -1     1004           1022       -18      B6    725  N804JB    JFK   BQN      183     1576     5     45 2013-01-01 05:00:00
# 5  2013     1     1      554            600        -6      812            837       -25      DL    461  N668DN    LGA   ATL      116      762     6      0 2013-01-01 06:00:00
# 6  2013     1     1      554            558        -4      740            728        12      UA   1696  N39463    EWR   ORD      150      719     5     58 2013-01-01 05:00:00
# 7  2013     1     1      555            600        -5      913            854        19      B6    507  N516JB    EWR   FLL      158     1065     6      0 2013-01-01 06:00:00
# 8  2013     1     1      557            600        -3      709            723       -14      EV   5708  N829AS    LGA   IAD       53      229     6      0 2013-01-01 06:00:00
# 9  2013     1     1      557            600        -3      838            846        -8      B6     79  N593JB    JFK   MCO      140      944     6      0 2013-01-01 06:00:00
# 10  2013     1     1      558            600        -2      753            745         8      AA    301  N3ALAA    LGA   ORD      138      733     6      0 2013-01-01 06:00:00
# # ... with 3.368e+05 more rows

# another option is using the View() function - which will give you the full dataset as a object in RStudio's built in viewer

## SUBSETTING
# all the tools we learned have worked with complete data frames
# if you want to pull out a single variable, we need some new tools: $, [[

# [[  can extract by name or position; $ extracts by name
df <- dplyr::tibble(
        x = runif(5),
        y = rnorm(5)
)

# extract by name
df$x
# [1] 0.4519528 0.7141528 0.8742177 0.8827714 0.5488999

# extract by position
df[[1]]
# [1] 0.4519528 0.7141528 0.8742177 0.8827714 0.5488999


# compared to data.frame, tibbles are more strict: they never do partial matching and will generate warning if column does not exist


## Interacting with older code...
# some older functions do not work with tibbles...
# if you encounter one of these functions use as.data.frame() to change a tibble back to a data frame
class(as.data.frame(tb))
# [1] "data.frame"






# chapter 8 import data ---------------------------------------------------



## Chapter 8: import with readr
# we will learn how to import data into R with readR
install.packages("readr")
library(readr)

# most of readr functions are concerned with turning flat files into data frames
        # read_csv() reads comma-delimited files
        # read_csv2() reads semicolon-deliminated files
        # read_tsv() reads tab-delimited files
        # read_delim() reads in file with any delimiter
        # read_fwf() reads fixed width files
        # read_log() reads Apache style log files
# all these functions have similiar functions: once you have mastered one you can use them all

# the first argument to read_csv is the most important: it is the path to the file to read
# you need to have this data sheet available in your working directory
heights <- read_csv("data/heights.csv")

# you can also read in a inline .csv file
read_csv(
        "a,b,c
        1,2,3
        4,5,6"
)

# # A tibble: 2 x 3
# a     b     c
# <int> <int> <int>
#         1     1     2     3
# 2     4     5     6

# in both cases above we use the first line of the data for the column names: you can tweak this
read_csv("The first line of metadata
         The second line of metadata
         x,y,z
         1,2,3", skip = 2)

# # A tibble: 1 x 3
# x     y     z
# <int> <int> <int>
#         1     1     2     3

# data might not have column names - you can use col_names = FALSE
read_csv("1,2,3\n4, 5,6", col_names = F)

# # A tibble: 2 x 3
# X1    X2    X3
# <int> <int> <int>
#         1     1     2     3
# 2     4     5     6


# \n is a shortcut to add a new line to your inline csv


# you can also pass column names you want into the csv as a character vector
read_csv("1,2,3\n4,5,6", col_names = c("x","y","z"))

# # A tibble: 2 x 3
# x     y     z
# <int> <int> <int>
#         1     1     2     3
# 2     4     5     6


# another option that needs tweaking is na()
# this specifies the values we want to change NA values into
read_csv("a,b,c\n1,2,.", na = ".")

# # A tibble: 1 x 3
# a     b     c
# <int> <int> <chr>
#         1     1     2  <NA>


## why do we favor readr over base R?
        # readr functions are typically must faster than thier base counterpart
        # readr functions produce tibbles: they don't convert character vectors to factors, use row names, or munge column names
        # they are more reproducible: base R inherits behavior from your operating system and your enviroment


# Parsing a Vector
# parse functions take a character vector and return a more specialized vector like logical, integer or date
str(parse_logical(c("TRUE","FALSE","NA")))
# logi [1:3] TRUE FALSE NA
str(parse_integer(c("1","2","3")))
# int [1:3] 1 2 3
str(parse_date(c("2010-01-01","1979-10-14")))
# Date[1:2], format: "2010-01-01" "1979-10-14"

# these functions are useful and are the building blocks of readr
# all parse functions are uniform
# first agrument is a character to parse, then the na arguement regarding which strings to count as NA
parse_integer(c("1","231",".","456"), na = ".")
# [1]   1 231  NA 456

# if parsing fails you'll get a warnings
x <- parse_integer(c("123","345","abc","123.45"))
# Warning: 2 parsing failures.
# row # A tibble: 2 x 4 col     row   col               expected actual expected   <int> <int>                  <chr>  <chr> actual 1     3    NA             an integer    abc row 2     4    NA no trailing characters    .45
# 
# Warning message:
#         In rbind(names(probs), probs_f) :
#         number of columns of result is not a multiple of vector length (arg 1)

# if there are any parsing issues, you'll need to run problems() to get the complete set
# this returns a tibble which you can manipulate with dplyr

# there are many parse functions available
        # parse_logical
        # parse_integer
        # parse_double = strict numeric parser
        # parse_number = flexible numeric parser
        # parse_character 
        # parse_factor
        # parse_datetime, parse_date, parse_time




## Parsing Numbers
# people write numbers differently - grouping characters or rounded numbers
# parsing numbers is based on decimal place which is then based on system local!!
parse_double("1.23")
# [1] 1.23
parse_double("1,23", locale = locale(decimal_mark = ","))
# [1] 1.23

# parse_number ignores non-numeric characters in the number "String"
parse_number("$100")
# [1] 100
parse_number("20%")
# [1] 20
parse_number("It cost $123.45")
# [1] 123.45


## Parsing Strings
# parsing strings is equally challenging
# R representing strings in interesting ways: as numbers!
charToRaw("Olivier")
# [1] 4f 6c 69 76 69 65 72

# each hexidecimal represents a byte of information - this mapping from the decimal to the string is called encoding!
# case encoding is called ASCII = the American Standard Code for Information Interchange
# today there is one standard supported almost everywhere: UTF-8

# readr uses UTF-8 everywhere; assumes UTF-8 encoding
# strings will look weird if there is not UTF-8 encoding
x1 <- "el ni\xf1o was particulary bad this year"
x2 <- "\x82\xb1\x82"

# to fix this call parse_character and set the correct encoding
# you can also have readr guess the encoding
parse_character(x1, locale = locale(encoding = "Latin1"))
# [1] "el niño was particulary bad this year"
parse_character(x2, locale = locale(encoding = "Shift-JIS"))

guess_encoding(charToRaw(x1))
# # A tibble: 2 x 2
# encoding confidence
# <chr>      <dbl>
#         1 ISO-8859-1       0.47
# 2 ISO-8859-9       0.23

guess_encoding((charToRaw(x2)))
# # A tibble: 1 x 2
# encoding confidence
# <chr>      <int>
#         1     <NA>         NA


## Factors
# R uses factors to represent categorical variables that have a known set of values
# if you have issues with the factor parse levels...parse as a character
fruit <- c("apple","banana")
parse_factor(c("apple","banana","bananana"), levels = fruit)
# Warning: 1 parsing failure.
# row # A tibble: 1 x 4 col     row   col           expected   actual expected   <int> <int>              <chr>    <chr> actual 1     3    NA value in level set bananana
# 
# [1] apple  banana <NA>  
#         attr(,"problems")
# # A tibble: 1 x 4
# row   col           expected   actual
# <int> <int>              <chr>    <chr>
#         1     3    NA value in level set bananana
# Levels: apple banana


## Dates, Date-Times, Times
# you pick between three parsers depending on whether you want a date, date time, or time
# parse date time expects ISO8601 date time = YEAR MONTH DAY HOUR MINUTE SECOND = biggest to smallest standard
parse_datetime("2010-10-01T2010")
# [1] "2010-10-01 20:10:00 UTC"
parse_datetime("20101010")
# [1] "2010-10-10 UTC"

# parse date expects four digit year, seperator as "-" or "/", the month, then the day
parse_date("2010-10-01")
# [1] "2010-10-01"

# parse_time expects the hour, :, minutes, optionally : and seconds
library(hms)
parse_time("01:10 am")
# 01:10:00
parse_time("20:10:01")
# 20:10:01

# if the defaults do not work for you you can supply your own date-time format, built up from the following peices
# YEAR = %Y (four digits), %y (two digits)
# MONTH = %m (two digits), %b (abbreviated name), %B (full name)
# DAY = %d (two digits), %e (optional leadings space)
# TIME = %H (0-23 hour format), %I (0-12, must be used with %p), %p (am / pm indicator), %M (minutes), %S (integer seconds), %OS (real seconds)

# the best way to figure out the needed format is to create a few examples and then test the parsing functions
parse_date("01/02/15", "%m/%d/%y")
# [1] "2015-01-02"
parse_date("01/02/15", "%d/%m/%y")
# [1] "2015-02-01"
parse_date("01/02/15", "%y/%m/%d")
# [1] "2001-02-15"



## Parsing Files
# these parse functions are how readr parses whole files!!!
# readr will automatically guess the type of each column
# we can override the default specification

# strategy: readr uses heuristics to figure out the type of each column:
# readr will read the first 1000 rows and uses some hueristics to figure out each column
# this can be emulated using the guess_parser() which returns readr's best guess
# use parse_guess() to use the best guess to parse the file's columns
guess_parser("2010-10-01")
# [1] "date"
guess_parser("15:01")
# [1] "time"
guess_parser(c("TRUE","FALSE"))
# [1] "logical"
guess_parser(c("1","2","3"))
# [1] "integer"
guess_parser(c("12,352,561"))
# [1] "number"
str(parse_guess("2010-10-10"))
# Date[1:1], format: "2010-10-10"

# readr tries a heuristic for each type and stops when it finds a match
# if none of the heuristics find a match - readr will return a vector
        # logical = contains only F, T, FALSE, TRUE
        # interger = contains only numeric and -
        # double = contains only valid doubles
        # number = contains valid doubles with grouping marks
        # time = match the default time_format
        # date = matches the default date_format
        # date-time = any ISO8601 date

## PROBLEMS
# readr contains a test .csv to exemplify some of the problems with the guessing
challenge <- read_csv(readr_example("challenge.csv"))

# Parsed with column specification:
#         cols(
#                 x = col_integer(),
#                 y = col_character()
#         )
# Warning: 1000 parsing failures.
# row # A tibble: 5 x 5 col     row   col               expected             actual expected   <int> <chr>                  <chr>              <chr> actual 1  1001     x no trailing characters .23837975086644292 file 2  1002     x no trailing characters .41167997173033655 row 3  1003     x no trailing characters  .7460716762579978 col 4  1004     x no trailing characters   .723450553836301 expected 5  1005     x no trailing characters   .614524137461558 actual # ... with 1 more variables: file <chr>
# ... ................. ... ....................................................... ........ ....................................................... ...... ....................................................... .... ....................................................... ... ....................................................... ... ....................................................... ........ ....................................................... ...... ........... [... truncated]
# Warning message:
#         In rbind(names(probs), probs_f) :
#         number of columns of result is not a multiple of vector length (arg 1)

# there are two printed outputs: the column specification generated by looking at the first 1000 rows, and the first five parsing failures
# it is always a good idea to check explicitly with problems()
problems(challenge)
# # A tibble: 1,000 x 5
# row   col               expected             actual
# <int> <chr>                  <chr>              <chr>
#         1  1001     x no trailing characters .23837975086644292
# 2  1002     x no trailing characters .41167997173033655
# 3  1003     x no trailing characters  .7460716762579978
# 4  1004     x no trailing characters   .723450553836301
# 5  1005     x no trailing characters   .614524137461558
# 6  1006     x no trailing characters   .473980569280684
# 7  1007     x no trailing characters  .5784610391128808
# 8  1008     x no trailing characters  .2415937229525298
# 9  1009     x no trailing characters .11437866208143532
# 10  1010     x no trailing characters  .2983446326106787
# # ... with 990 more rows, and 1 more variables: file <chr>


# a good strategy to solve this is to work column by column until there are no parsing errors
challenge <- read_csv(
        readr_example("challenge.csv"),
        col_types = cols(
                x = col_double(),
                y = col_character()
        )
)

# this fixes the first problem ^^^^

tail(challenge)
# # A tibble: 6 x 2
# x          y
# <dbl>      <chr>
#         1 0.8052743 2019-11-21
# 2 0.1635163 2018-03-29
# 3 0.4719390 2014-08-04
# 4 0.7183186 2015-08-16
# 5 0.2698786 2020-02-04
# 6 0.6082372 2019-01-06

# need to fix the date column now...
challenge <- read_csv(
        readr_example("challenge.csv"),
        col_types = cols(
                x = col_double(),
                y = col_date()
        )
)

tail(challenge)
# # A tibble: 6 x 2
# x          y
# <dbl>     <date>
#         1 0.8052743 2019-11-21
# 2 0.1635163 2018-03-29
# 3 0.4719390 2014-08-04
# 4 0.7183186 2015-08-16
# 5 0.2698786 2020-02-04
# 6 0.6082372 2019-01-06


# every parse function has a corresponding col_types function
# you can use the col_types function to tell readr how you want to read the file in 

# we can also change the default to look past the first 1000 rows
challenge2 <- read_csv(
        readr_example("challenge.csv"),
        guess_max = 1001
)
# Parsed with column specification:
#         cols(
#                 x = col_double(),
#                 y = col_date(format = "")
#         )
challenge2
# # A tibble: 2,000 x 2
# x      y
# <dbl> <date>
#         1   404     NA
# 2  4172     NA
# 3  3004     NA
# 4   787     NA
# 5    37     NA
# 6  2332     NA
# 7  2489     NA
# 8  1449     NA
# 9  3665     NA
# 10  3863     NA
# # ... with 1,990 more rows


# sometimes it is easier to diagnose the problem by reading in all columns are character
challenge2 <- read_csv(readr_example("challenge.csv"),
                       col_types = cols(.default = col_character()))
# you can then use type_convert to guess back at the correct type of parsing
df <- tribble(
        ~x, ~y,
        "1", "1.21",
        "2", "2.32"
)
df
# # A tibble: 2 x 2
# x     y
# <chr> <chr>
#         1     1  1.21
# 2     2  2.32

type_convert(df)
# # A tibble: 2 x 2
# x     y
# <int> <dbl>
#         1     1  1.21
# 2     2  2.32

# if we are readining in a very large file, set n_max to a smallish number 10,000 or 100,000
# this will excelerate your iterations while you eliminate common problems


## Writing a file
# readr also comes with two useful functions for writing data back to disk: write_csv and write_tsv
# these always are UTF-8 encoded
# saves dates and date-times in ISO8601 format so they are easily parsed everywhere
# if you want to export a CSV into EXcel, use write_excel_csv
# the most important arguements are x (the data frame to save) and path ( the location to save it)
write_csv(challenge, "challenge.csv")

# note that the type information is lost when you save to CSV
challenge
# # A tibble: 2,000 x 2
# x      y
# <dbl> <date>
#         1   404     NA
# 2  4172     NA
# 3  3004     NA
# 4   787     NA
# 5    37     NA
# 6  2332     NA
# 7  2489     NA
# 8  1449     NA
# 9  3665     NA
# 10  3863     NA
# # ... with 1,990 more rows

write_csv(challenge, "challenge-2.csv")
read_csv("challenge-2.csv")

# Parsed with column specification:
#         cols(
#                 x = col_integer(),
#                 y = col_character()
#         )
# Warning: 1000 parsing failures.
# row # A tibble: 5 x 5 col     row   col               expected             actual              file expected   <int> <chr>                  <chr>              <chr>             <chr> actual 1  1001     x no trailing characters .23837975086644292 'challenge-2.csv' file 2  1002     x no trailing characters .41167997173033655 'challenge-2.csv' row 3  1003     x no trailing characters  .7460716762579978 'challenge-2.csv' col 4  1004     x no trailing characters   .723450553836301 'challenge-2.csv' expected 5  1005     x no trailing characters   .614524137461558 'challenge-2.csv'
# ... ................. ... ......................................................................... ........ ......................................................................... ...... ......................................................................... .... ......................................................................... ... ................................................ [... truncated]
# # A tibble: 2,000 x 2
# x     y
# <int> <chr>
#         1   404  <NA>
#         2  4172  <NA>
#         3  3004  <NA>
#         4   787  <NA>
#         5    37  <NA>
#         6  2332  <NA>
#         7  2489  <NA>
#         8  1449  <NA>
#         9  3665  <NA>
#         10  3863  <NA>
#         # ... with 1,990 more rows
#         Warning message:
#         In rbind(names(probs), probs_f) :
#         number of columns of result is not a multiple of vector length (arg 1)

# this makes csv a little hard to work with  - you need to re-create the column specifications every time you load it in
# alternatives: write_rdrs and read_rds are wrappers around base R functions that store data in R custom binary format RDS
write_rds(challenge, "challenge.rds")
read_rds("challenge.rds")
# # A tibble: 2,000 x 2
# x      y
# <dbl> <date>
#         1   404     NA
# 2  4172     NA
# 3  3004     NA
# 4   787     NA
# 5    37     NA
# 6  2332     NA
# 7  2489     NA
# 8  1449     NA
# 9  3665     NA
# 10  3863     NA
# # ... with 1,990 more rows

# the feather package implements a fast binary file format that can be shared across programming languages
install.packages("feather")
library(feather)
write_feather(challenge, "challenge.feather")
read_feather("challenge.feather")
# # A tibble: 2,000 x 2
# x      y
# <dbl> <date>
#         1   404     NA
# 2  4172     NA
# 3  3004     NA
# 4   787     NA
# 5    37     NA
# 6  2332     NA
# 7  2489     NA
# 8  1449     NA
# 9  3665     NA
# 10  3863     NA
# # ... with 1,990 more rows

# feather tends to be faster than RDS and is usable outside of R!!!!
# RDS supports list columns, feather does not!!!!

## to get other types of data into R: investigate the following tidyverse packages
        # haven = SPSS, Stata, SAS
        # readxl = read excel files both .xls and .xlsx
        # DBI + RMYSQL, RSQLLite, RpostgreSQL = allows you to run SQL queries against a database and return a dataframe
        # jsonlite = JSON 
        # xml12 = XML
        # rio package = automatically guesses the best read or write type






# chapter 9 tidy data -----------------------------------------------------



##  Chapter 9: Tidy data with TidyR
# tidy data is a consistent way to organize your data in R
# tidy is work up front but will save you time later
# all these tools are available in the tidyr package
library(tidyr)

# you can represent underlying data in many different ways
# three rules to make a dataset tidy:
        # each variable must have its own column
        # each observation must have its own row
        # each value must have its own cell
# these rules are interrelated because it is impossible to only satisfy two of the three
        # put each dataset into a tibble
        # put each variable in a column
# why tidy?
        # if you have consistent data structure - its easier to learn the tools to manipulate that data
        # placing all variables in a column allows you to utilize R's vectorized nature - most columns use vectors of values
        # all other packages in tidyverse are designed for use with tidy data
table1 %>% 
        mutate(rate = cases / population * 10000)
# # A tibble: 6 x 5
# country  year  cases population     rate
# <chr> <int>  <int>      <int>    <dbl>
#         1 Afghanistan  1999    745   19987071 0.372741
# 2 Afghanistan  2000   2666   20595360 1.294466
# 3      Brazil  1999  37737  172006362 2.193930
# 4      Brazil  2000  80488  174504898 4.612363
# 5       China  1999 212258 1272915272 1.667495
# 6       China  2000 213766 1280428583 1.669488

table1 %>% 
        count(year, wt = cases)
# # A tibble: 2 x 2
# year      n
# <int>  <int>
#         1  1999 250740
# 2  2000 296920
library(ggplot2)
ggplot(table1, aes(year, cases)) +
        geom_line(aes(group = country), color = "grey50") +
        geom_point(aes(color = country))

## Spreading and Gathering
# most data you encounter will not be tidy
# we need functions to transform our untidy data to tidy
# most people do not use tidy data on thier own
# data is often organized to facilitate use other and analytics i.e. ease of entry
# for most real analysis - you will need to do some tidying up front
        # figure out what variables and observations are
        # resolve variables spread across columns
        # resolve obersvations scattered across multiple rows
# use gather() and spread() to solve these problems

## Gathering
# columns are not variables but values of a variable
# to tidy this dataset we need to gather these columns into one variable
# to do this we need:
        #the set of columns that represent values (the "bad" columns)
        # the name of the variable whose value forms the column names
        # the name of the variable whose values are spread over the cells
table4a
# # A tibble: 3 x 3
# country `1999` `2000`
# *       <chr>  <int>  <int>
#         1 Afghanistan    745   2666
# 2      Brazil  37737  80488
# 3       China 212258 213766


# this principals come together to define the gather() statement
table4a %>% 
        gather(`1999`,`2000`, key = "year", value = "cases")
# # A tibble: 6 x 3
# country  year  cases
# <chr> <chr>  <int>
#         1 Afghanistan  1999    745
# 2      Brazil  1999  37737
# 3       China  1999 212258
# 4 Afghanistan  2000   2666
# 5      Brazil  2000  80488
# 6       China  2000 213766

table4b %>% 
        gather(`1999`,`2000`, key = "year", value = "population")
# # A tibble: 6 x 3
# country  year population
# <chr> <chr>      <int>
#         1 Afghanistan  1999   19987071
# 2      Brazil  1999  172006362
# 3       China  1999 1272915272
# 4 Afghanistan  2000   20595360
# 5      Brazil  2000  174504898
# 6       China  2000 1280428583


# to combine the tidied versions of table4a and table4b into a single tibble;
# we need to use dplyr::left_join() - learn about this in relational databases
tidy4a = table4a %>% 
        gather(`1999`,`2000`, key = "year", value = "cases")
tidy4b = table4b %>% 
        gather(`1999`,`2000`, key = "year", value = "population")
left_join(tidy4a, tidy4b)
# # A tibble: 6 x 4
# country  year  cases population
# <chr> <chr>  <int>      <int>
#         1 Afghanistan  1999    745   19987071
# 2      Brazil  1999  37737  172006362
# 3       China  1999 212258 1272915272
# 4 Afghanistan  2000   2666   20595360
# 5      Brazil  2000  80488  174504898
# 6       China  2000 213766 1280428583


## Spreading
# spreading is the opposite of gathering
# use this when observations are scattered across multiple rows
table2
# # A tibble: 12 x 4
# country  year       type      count
# <chr> <int>      <chr>      <int>
#         1 Afghanistan  1999      cases        745
# 2 Afghanistan  1999 population   19987071
# 3 Afghanistan  2000      cases       2666
# 4 Afghanistan  2000 population   20595360
# 5      Brazil  1999      cases      37737
# 6      Brazil  1999 population  172006362
# 7      Brazil  2000      cases      80488
# 8      Brazil  2000 population  174504898
# 9       China  1999      cases     212258
# 10       China  1999 population 1272915272
# 11       China  2000      cases     213766
# 12       China  2000 population 1280428583


# to tidy this we:
        # the column that contains variable names, the key == type
        # the column that contains values from multiple variables, value == count
spread(table2, key = type, value = count)
# # A tibble: 6 x 4
# country  year  cases population
# *       <chr> <int>  <int>      <int>
#         1 Afghanistan  1999    745   19987071
# 2 Afghanistan  2000   2666   20595360
# 3      Brazil  1999  37737  172006362
# 4      Brazil  2000  80488  174504898
# 5       China  1999 212258 1272915272
# 6       China  2000 213766 1280428583

# spread and gather are compliments they have similiar key and value arguements
# gather = wide tables narrow
# spread = long tables shorter



## Seperating and Pull
# we have one column that contains two variables
# to fix this we need the seperate function
# seperate pulls apart one column into multiple columns
# seperate will split on the seperator indicator
table3
# # A tibble: 6 x 3
# country  year              rate
# *       <chr> <int>             <chr>
#         1 Afghanistan  1999      745/19987071
# 2 Afghanistan  2000     2666/20595360
# 3      Brazil  1999   37737/172006362
# 4      Brazil  2000   80488/174504898
# 5       China  1999 212258/1272915272
# 6       China  2000 213766/1280428583

# the rate column contains observations for both cases and population
        # takes the name of the column to seperate
        # the names of the columns to seperate into
table3 %>% 
        separate(rate, into = c("cases", "population"))
# # A tibble: 6 x 4
# country  year  cases population
# *       <chr> <int>  <chr>      <chr>
#         1 Afghanistan  1999    745   19987071
# 2 Afghanistan  2000   2666   20595360
# 3      Brazil  1999  37737  172006362
# 4      Brazil  2000  80488  174504898
# 5       China  1999 212258 1272915272
# 6       China  2000 213766 1280428583

# by default separate will split on the first non-alpha numeric character
# to split by a specific character use the sep function
table3 %>% 
        separate(rate, into = c("cases", "population"), sep = "/")
# # A tibble: 6 x 4
# country  year  cases population
# *       <chr> <int>  <chr>      <chr>
#         1 Afghanistan  1999    745   19987071
# 2 Afghanistan  2000   2666   20595360
# 3      Brazil  1999  37737  172006362
# 4      Brazil  2000  80488  174504898
# 5       China  1999 212258 1272915272
# 6       China  2000 213766 1280428583

# the default behavior of separte is character columns...
# to stop this use the convert = T arguement
table3 %>% 
        separate(
                rate,
                into = c("cases","population"),
                convert = T,
                sep = "/"
        )
# # A tibble: 6 x 4
# country  year  cases population
# *       <chr> <int>  <int>      <int>
#         1 Afghanistan  1999    745   19987071
# 2 Afghanistan  2000   2666   20595360
# 3      Brazil  1999  37737  172006362
# 4      Brazil  2000  80488  174504898
# 5       China  1999 212258 1272915272
# 6       China  2000 213766 1280428583


# you can also pass a vector of integers into sep
# this will split at the column specificied by the integer provided
table3 %>% 
        separate(year, into = c("century","year"), sep = 2)
# # A tibble: 6 x 4
# country century  year              rate
# *       <chr>   <chr> <chr>             <chr>
#         1 Afghanistan      19    99      745/19987071
# 2 Afghanistan      20    00     2666/20595360
# 3      Brazil      19    99   37737/172006362
# 4      Brazil      20    00   80488/174504898
# 5       China      19    99 212258/1272915272
# 6       China      20    00 213766/1280428583


## Unite
# unite is the inverse of separate: it combines multiple columns together into one
# we can use unite to rejoin the century and year columns from the last example
# unite takes the arguements:
        # a data frame
        # name of the new variable to create
        # a set of columns to combine (specified with dplyr::select)
table5 %>% 
        unite(new, century, year)
# # A tibble: 6 x 3
# country   new              rate
# *       <chr> <chr>             <chr>
#         1 Afghanistan 19_99      745/19987071
# 2 Afghanistan 20_00     2666/20595360
# 3      Brazil 19_99   37737/172006362
# 4      Brazil 20_00   80488/174504898
# 5       China 19_99 212258/1272915272
# 6       China 20_00 213766/1280428583

# the default sep arguement for unite is "_"
# use the sep function to give another type of separator
table5 %>% 
        unite(new, century, year, sep = "")
# # A tibble: 6 x 3
# country   new              rate
# *       <chr> <chr>             <chr>
#         1 Afghanistan  1999      745/19987071
# 2 Afghanistan  2000     2666/20595360
# 3      Brazil  1999   37737/172006362
# 4      Brazil  2000   80488/174504898
# 5       China  1999 212258/1272915272
# 6       China  2000 213766/1280428583


## Missing Values
# values can be missing in two possible ways:
        # explicitly: flagged with NA
        # implicitly: simply not present in the data

# example
# there are two missing values in this dataset
        # the return for fourth quarter 2015 is explictly missing = NA
        # the return for 1st quarter 2016 is implicitly missing, does not appear in dataset
stocks = tibble(
        year = c(2015, 2015, 2015, 2015, 2016, 2016, 2016),
        qtr = c(1, 2, 3, 4, 2, 3, 4),
        return = c(1.88,.59, .35, NA, .92, .17, 2.66)
)

# presence of an absence vs. absense of a presensce
# one way you can handle this is to put each year in a column == NOT TIDY!!
stocks %>% 
        spread(year, return)
# # A tibble: 4 x 3
# qtr `2015` `2016`
# * <dbl>  <dbl>  <dbl>
#         1     1   1.88     NA
# 2     2   0.59   0.92
# 3     3   0.35   0.17
# 4     4     NA   2.66

# becuase explicit missing values may not be important in other representations of the data
# set na.rm = T in the gather() statement = turns explicit into implicit
stocks %>% 
        spread(year, return) %>% 
        gather(year, return, `2015`:`2016`, na.rm = T)
# # A tibble: 6 x 3
# qtr  year return
# * <dbl> <chr>  <dbl>
#         1     1  2015   1.88
# 2     2  2015   0.59
# 3     3  2015   0.35
# 4     2  2016   0.92
# 5     3  2016   0.17
# 6     4  2016   2.66

# another option for making missing values explicit is complete()
# complete takes a set of columns and finds all unique combinations
# then ensures the original dataset contains all those values, filling in NA where missing
stocks %>% 
        complete(year, qtr)
# # A tibble: 8 x 3
# year   qtr return
# <dbl> <dbl>  <dbl>
#         1  2015     1   1.88
# 2  2015     2   0.59
# 3  2015     3   0.35
# 4  2015     4     NA
# 5  2016     1     NA
# 6  2016     2   0.92
# 7  2016     3   0.17
# 8  2016     4   2.66



# yet another option is the fill() arguement
# sometimes NAs mean the previous value should be filled in or carried forward:
# you can fill in the missing values with the most recent non-missing value
treatment = tribble(
        ~person, ~treatment, ~repsonse,
        "derrick", 1, 7,
        NA, 2, 10,
        NA, 3, 9,
        "katherine", 1, 4
)
# # A tibble: 4 x 3
# person treatment repsonse
# <chr>     <dbl>    <dbl>
#         1   derrick         1        7
# 2      <NA>         2       10
# 3      <NA>         3        9
# 4 katherine         1        4

treatment %>% 
        fill(person)
# 
# # A tibble: 4 x 3
# person treatment repsonse
# <chr>     <dbl>    <dbl>
#         1   derrick         1        7
# 2   derrick         2       10
# 3   derrick         3        9
# 4 katherine         1        4




## CASE STUDY
# let's put all tidy data concepts together into one test case
# let's use the real life dataset from tidyR: who
# it contains redundant columns, odd variable codes, missing values
# this data is messy and we need to tidy it!!!
# we will need to string together multiple tidy verbs to completely clean the dataset
data(who)
who

# # A tibble: 7,240 x 60
# country  iso2  iso3  year new_sp_m014 new_sp_m1524 new_sp_m2534
# <chr> <chr> <chr> <int>       <int>        <int>        <int>
#         1 Afghanistan    AF   AFG  1980          NA           NA           NA
# 2 Afghanistan    AF   AFG  1981          NA           NA           NA
# 3 Afghanistan    AF   AFG  1982          NA           NA           NA
# 4 Afghanistan    AF   AFG  1983          NA           NA           NA
# 5 Afghanistan    AF   AFG  1984          NA           NA           NA
# 6 Afghanistan    AF   AFG  1985          NA           NA           NA
# 7 Afghanistan    AF   AFG  1986          NA           NA           NA
# 8 Afghanistan    AF   AFG  1987          NA           NA           NA
# 9 Afghanistan    AF   AFG  1988          NA           NA           NA
# 10 Afghanistan    AF   AFG  1989          NA           NA           NA
# # ... with 7,230 more rows, and 53 more variables: new_sp_m3544 <int>,
# #   new_sp_m4554 <int>, new_sp_m5564 <int>, new_sp_m65 <int>,
# #   new_sp_f014 <int>, new_sp_f1524 <int>, new_sp_f2534 <int>,
# #   new_sp_f3544 <int>, new_sp_f4554 <int>, new_sp_f5564 <int>,
# #   new_sp_f65 <int>, new_sn_m014 <int>, new_sn_m1524 <int>,
# #   new_sn_m2534 <int>, new_sn_m3544 <int>, new_sn_m4554 <int>,
# #   new_sn_m5564 <int>, new_sn_m65 <int>, new_sn_f014 <int>,
# #   new_sn_f1524 <int>, new_sn_f2534 <int>, new_sn_f3544 <int>,
# #   new_sn_f4554 <int>, new_sn_f5564 <int>, new_sn_f65 <int>,
# #   new_ep_m014 <int>, new_ep_m1524 <int>, new_ep_m2534 <int>,
# #   new_ep_m3544 <int>, new_ep_m4554 <int>, new_ep_m5564 <int>,
# #   new_ep_m65 <int>, new_ep_f014 <int>, new_ep_f1524 <int>,
# #   new_ep_f2534 <int>, new_ep_f3544 <int>, new_ep_f4554 <int>,
# #   new_ep_f5564 <int>, new_ep_f65 <int>, newrel_m014 <int>,
# #   newrel_m1524 <int>, newrel_m2534 <int>, newrel_m3544 <int>,
# #   newrel_m4554 <int>, newrel_m5564 <int>, newrel_m65 <int>,
# #   newrel_f014 <int>, newrel_f1524 <int>, newrel_f2534 <int>,
# #   newrel_f3544 <int>, newrel_f4554 <int>, newrel_f5564 <int>,
# #   newrel_f65 <int>


# the best place to start is to gather together all columns that are not variables
# country, iso2, iso3 are three variables that redundantly specify country
# year is also a variable
# other columns are likely to be values not variables new_sp_m014 etc.
# let's gather all these column with a generic key, variable cases, na.rm = T
who1 <- who %>% 
        gather(
                new_sp_m014:newrel_f65, key = "key",
                value = "cases",
                na.rm = T
        )
who1

# # A tibble: 76,046 x 6
# country  iso2  iso3  year         key cases
# *       <chr> <chr> <chr> <int>       <chr> <int>
#         1 Afghanistan    AF   AFG  1997 new_sp_m014     0
# 2 Afghanistan    AF   AFG  1998 new_sp_m014    30
# 3 Afghanistan    AF   AFG  1999 new_sp_m014     8
# 4 Afghanistan    AF   AFG  2000 new_sp_m014    52
# 5 Afghanistan    AF   AFG  2001 new_sp_m014   129
# 6 Afghanistan    AF   AFG  2002 new_sp_m014    90
# 7 Afghanistan    AF   AFG  2003 new_sp_m014   127
# 8 Afghanistan    AF   AFG  2004 new_sp_m014   139
# 9 Afghanistan    AF   AFG  2005 new_sp_m014   151
# 10 Afghanistan    AF   AFG  2006 new_sp_m014   193
# # ... with 76,036 more rows

# we get a hint of the stucture of the value in the new key column by counting them
who1 %>% count(key)
# # A tibble: 56 x 2
# key     n
# <chr> <int>
#         1  new_ep_f014  1032
# 2 new_ep_f1524  1021
# 3 new_ep_f2534  1021
# 4 new_ep_f3544  1021
# 5 new_ep_f4554  1017
# 6 new_ep_f5564  1017
# 7   new_ep_f65  1014
# 8  new_ep_m014  1038
# 9 new_ep_m1524  1026
# 10 new_ep_m2534  1020
# # ... with 46 more rows

# from the data dictionary:
# the first three letters of each column denote whether the column contains new or old cases
# each column contains new cases
# the next two letters describe the type of TB
        # rel = relapse
        # ep = extrapulmonary TB
        # sn = diagnosis by smear negative
        # sp = diagnosis by smear positive
# the sixth letter gives the sex of the patients
# the remaining numbers give the age group


# let's fix some names
(who2 <- who1 %>% 
        mutate(key = stringr::str_replace(key, "newrl","new_rel")))

# # A tibble: 76,046 x 6
# country  iso2  iso3  year         key cases
# <chr> <chr> <chr> <int>       <chr> <int>
#         1 Afghanistan    AF   AFG  1997 new_sp_m014     0
# 2 Afghanistan    AF   AFG  1998 new_sp_m014    30
# 3 Afghanistan    AF   AFG  1999 new_sp_m014     8
# 4 Afghanistan    AF   AFG  2000 new_sp_m014    52
# 5 Afghanistan    AF   AFG  2001 new_sp_m014   129
# 6 Afghanistan    AF   AFG  2002 new_sp_m014    90
# 7 Afghanistan    AF   AFG  2003 new_sp_m014   127
# 8 Afghanistan    AF   AFG  2004 new_sp_m014   139
# 9 Afghanistan    AF   AFG  2005 new_sp_m014   151
# 10 Afghanistan    AF   AFG  2006 new_sp_m014   193
# # ... with 76,036 more rows

# we can separate each values of the code with separate
(who3 <- who2 %>% 
        separate(key, c("new","type","sexage"), sep = "_"))

# # A tibble: 76,046 x 8
# country  iso2  iso3  year   new  type sexage cases
# *       <chr> <chr> <chr> <int> <chr> <chr>  <chr> <int>
#         1 Afghanistan    AF   AFG  1997   new    sp   m014     0
# 2 Afghanistan    AF   AFG  1998   new    sp   m014    30
# 3 Afghanistan    AF   AFG  1999   new    sp   m014     8
# 4 Afghanistan    AF   AFG  2000   new    sp   m014    52
# 5 Afghanistan    AF   AFG  2001   new    sp   m014   129
# 6 Afghanistan    AF   AFG  2002   new    sp   m014    90
# 7 Afghanistan    AF   AFG  2003   new    sp   m014   127
# 8 Afghanistan    AF   AFG  2004   new    sp   m014   139
# 9 Afghanistan    AF   AFG  2005   new    sp   m014   151
# 10 Afghanistan    AF   AFG  2006   new    sp   m014   193
# # ... with 76,036 more rows


# lets drop the redundant columns
who3 %>% count(new)

# # A tibble: 2 x 2
# new     n
# <chr> <int>
#         1    new 73466
# 2 newrel  2580

who4 <- who3 %>% 
        dplyr::select(., -new,-iso2,-iso3)

# next we'll seprate sex and age by splitting after the first character
(who5 <- who4 %>% 
        separate(sexage, c("sex","age"), sep = 1))

# # A tibble: 76,046 x 6
# country  year  type   sex   age cases
# *       <chr> <int> <chr> <chr> <chr> <int>
#         1 Afghanistan  1997    sp     m   014     0
# 2 Afghanistan  1998    sp     m   014    30
# 3 Afghanistan  1999    sp     m   014     8
# 4 Afghanistan  2000    sp     m   014    52
# 5 Afghanistan  2001    sp     m   014   129
# 6 Afghanistan  2002    sp     m   014    90
# 7 Afghanistan  2003    sp     m   014   127
# 8 Afghanistan  2004    sp     m   014   139
# 9 Afghanistan  2005    sp     m   014   151
# 10 Afghanistan  2006    sp     m   014   193
# # ... with 76,036 more rows


## THIS IS NOW A TIDY DATASET!!!
# all code in one fell swoop
(who %>% 
        gather(code, value, new_sp_m014:newrel_f65, na.rm = T) %>% 
        mutate(code = stringr::str_replace(code, "newrel","new_rel")) %>% 
        separate(code,c("new","var","sexage")) %>% 
        dplyr::select(., -new, -iso2, -iso3) %>% 
        separate(sexage, c("sex","age"), sep = 1))

# # A tibble: 76,046 x 6
# country  year   var   sex   age value
# *       <chr> <int> <chr> <chr> <chr> <int>
#         1 Afghanistan  1997    sp     m   014     0
# 2 Afghanistan  1998    sp     m   014    30
# 3 Afghanistan  1999    sp     m   014     8
# 4 Afghanistan  2000    sp     m   014    52
# 5 Afghanistan  2001    sp     m   014   129
# 6 Afghanistan  2002    sp     m   014    90
# 7 Afghanistan  2003    sp     m   014   127
# 8 Afghanistan  2004    sp     m   014   139
# 9 Afghanistan  2005    sp     m   014   151
# 10 Afghanistan  2006    sp     m   014   193
# # ... with 76,036 more rows


## non tidy data
# the are useful and well-founded data structures that are not tidy data
# reasons to use non-tidy data
        # alternative representations may have substantial performance / space advantages
        # specialized fields have evolved thier own conventions for storing data 
# if your data is in this format you'll need something other than tibble or data frame
# TIDY SHOULD BE YOUR DEFAULT CHOICE









# chapter 10 relational data ----------------------------------------------



## Chapter 10: Relational Data with dplyr
# mutiple tables of data are called relational data
# the relationship between datasets are important
# each table must have a relation key to map to the other tables
# you need some dplyr verbs to work with multiple relational tables
        # mutating joins - add new variables to one frame from matching obs in another
        # filtering joins - filter obs from one data frame based on a match in another
        # set operations - treats operations as if they were set elements
# most relationship data are found in relational database management system
# this encompasses most modern databases
# commonly use SQL to work with relational databases
# dplyr will be similiar to SQL but specialized to do data analysis

# let's use the four tables in the nyc flights data to explore
library(nycflights13)

airlines
# # A tibble: 16 x 2
# carrier                        name
# <chr>                       <chr>
#         1      9E           Endeavor Air Inc.
# 2      AA      American Airlines Inc.
# 3      AS        Alaska Airlines Inc.
# 4      B6             JetBlue Airways
# 5      DL        Delta Air Lines Inc.
# 6      EV    ExpressJet Airlines Inc.
# 7      F9      Frontier Airlines Inc.
# 8      FL AirTran Airways Corporation
# 9      HA      Hawaiian Airlines Inc.
# 10      MQ                   Envoy Air
# 11      OO       SkyWest Airlines Inc.
# 12      UA       United Air Lines Inc.
# 13      US             US Airways Inc.
# 14      VX              Virgin America
# 15      WN      Southwest Airlines Co.
# 16      YV          Mesa Airlines Inc.


airports
# # A tibble: 1,458 x 8
# faa                           name      lat        lon   alt    tz
# <chr>                          <chr>    <dbl>      <dbl> <int> <dbl>
#         1   04G              Lansdowne Airport 41.13047  -80.61958  1044    -5
# 2   06A  Moton Field Municipal Airport 32.46057  -85.68003   264    -6
# 3   06C            Schaumburg Regional 41.98934  -88.10124   801    -6
# 4   06N                Randall Airport 41.43191  -74.39156   523    -5
# 5   09J          Jekyll Island Airport 31.07447  -81.42778    11    -5
# 6   0A9 Elizabethton Municipal Airport 36.37122  -82.17342  1593    -5
# 7   0G6        Williams County Airport 41.46731  -84.50678   730    -5
# 8   0G7  Finger Lakes Regional Airport 42.88356  -76.78123   492    -5
# 9   0P2   Shoestring Aviation Airfield 39.79482  -76.64719  1000    -5
# 10   0S9          Jefferson County Intl 48.05381 -122.81064   108    -8
# # ... with 1,448 more rows, and 2 more variables: dst <chr>,
# #   tzone <chr>

planes
# A tibble: 3,322 x 9
# tailnum  year                    type     manufacturer     model
# <chr> <int>                   <chr>            <chr>     <chr>
#         1  N10156  2004 Fixed wing multi engine          EMBRAER EMB-145XR
# 2  N102UW  1998 Fixed wing multi engine AIRBUS INDUSTRIE  A320-214
# 3  N103US  1999 Fixed wing multi engine AIRBUS INDUSTRIE  A320-214
# 4  N104UW  1999 Fixed wing multi engine AIRBUS INDUSTRIE  A320-214
# 5  N10575  2002 Fixed wing multi engine          EMBRAER EMB-145LR
# 6  N105UW  1999 Fixed wing multi engine AIRBUS INDUSTRIE  A320-214
# 7  N107US  1999 Fixed wing multi engine AIRBUS INDUSTRIE  A320-214
# 8  N108UW  1999 Fixed wing multi engine AIRBUS INDUSTRIE  A320-214
# 9  N109UW  1999 Fixed wing multi engine AIRBUS INDUSTRIE  A320-214
# 10  N110UW  1999 Fixed wing multi engine AIRBUS INDUSTRIE  A320-214
# # ... with 3,312 more rows, and 4 more variables: engines <int>,
# #   seats <int>, speed <int>, engine <chr>


weather
# # A tibble: 26,130 x 15
# origin  year month   day  hour  temp  dewp humid wind_dir wind_speed
# <chr> <dbl> <dbl> <int> <int> <dbl> <dbl> <dbl>    <dbl>      <dbl>
#         1    EWR  2013     1     1     0 37.04 21.92 53.97      230   10.35702
# 2    EWR  2013     1     1     1 37.04 21.92 53.97      230   13.80936
# 3    EWR  2013     1     1     2 37.94 21.92 52.09      230   12.65858
# 4    EWR  2013     1     1     3 37.94 23.00 54.51      230   13.80936
# 5    EWR  2013     1     1     4 37.94 24.08 57.04      240   14.96014
# 6    EWR  2013     1     1     6 39.02 26.06 59.37      270   10.35702
# 7    EWR  2013     1     1     7 39.02 26.96 61.63      250    8.05546
# 8    EWR  2013     1     1     8 39.02 28.04 64.43      240   11.50780
# 9    EWR  2013     1     1     9 39.92 28.04 62.21      250   12.65858
# 10    EWR  2013     1     1    10 39.02 28.04 64.43      260   12.65858
# # ... with 26,120 more rows, and 5 more variables: wind_gust <dbl>,
# #   precip <dbl>, pressure <dbl>, visib <dbl>, time_hour <dttm>


# each relation will always concern a pair of tables
# we need to find the chain linking each table to each other

# flights connects to planes by tailnum
# flights connects to airlines through the carrier variable
# flights connects to airports in two ways: origin and dest variables
# flights connects to weather via origina and year, month, day, hour


## Keys
# the variables used to connect each pair of tables are called keys
# a key uniques identifies an observation
# example: plane is identified by tailnum
# sometimes you may need multiple keys to join tables together
# there are two types of keys:
        # primary key: identifies obs in its own table
        # foriegn key: identifies obs in another table
# a variable can be both a primary key and a foriegn key
# once you identify the primary keys you can check if they are primary keys
# count the keys and look for entries where n is greater than one

# no values will mean all the keys are primary
planes %>% 
        count(tailnum) %>% 
        filter(n>1)
# A tibble: 0 x 2
# ... with 2 variables: tailnum <chr>, n <int>

weather %>% 
        count(year, month, day, hour, origin) %>% 
        filter(n>1)
# A tibble: 0 x 6
# ... with 6 variables: year <dbl>, month <dbl>, day <int>, hour <int>,
#   origin <chr>, n <int>

# sometimes a table doesn't have an explicit primary key
# each row is an observation, but no combination of variables uniquely identifies it

# are there primary keys in flights?
flights %>% 
        count(year, month, day, flight) %>% 
        filter(n>1)
# # A tibble: 29,768 x 5
# year month   day flight     n
# <int> <int> <int>  <int> <int>
#         1  2013     1     1      1     2
# 2  2013     1     1      3     2
# 3  2013     1     1      4     2
# 4  2013     1     1     11     3
# 5  2013     1     1     15     2
# 6  2013     1     1     21     2
# 7  2013     1     1     27     4
# 8  2013     1     1     31     2
# 9  2013     1     1     32     2
# 10  2013     1     1     35     2
# # ... with 29,758 more rows

flights %>%
        count(year, month, day, tailnum) %>% 
        filter(n > 1)
# # A tibble: 64,928 x 5
# year month   day tailnum     n
# <int> <int> <int>   <chr> <int>
#         1  2013     1     1  N0EGMQ     2
# 2  2013     1     1  N11189     2
# 3  2013     1     1  N11536     2
# 4  2013     1     1  N11544     3
# 5  2013     1     1  N11551     2
# 6  2013     1     1  N12540     2
# 7  2013     1     1  N12567     2
# 8  2013     1     1  N13123     2
# 9  2013     1     1  N13538     3
# 10  2013     1     1  N13566     3
# # ... with 64,918 more rows

# we thought each flights number would only be used once per day
# THIS IS NOT THE CASE!
# flights table lacks a primary key
# we can add a surrogate key with mutate and row_number
# a primary key and the foriegn key in another table form a relation
# relationships are typically one to many
# each flight may have one plane, but a plane has many flights
# you may also see a 1-to-1 relationship
# you can model many-to-many with a many-to-1 relation plus a 1-to-many relation


## Mutating Joins
# this is the first tool we will investigate
# mutating joins allow you to combine variables from two tables
# it first matches observations by thier keys, then copies across variables 
# new variables are added to the right of the joined data set
(flights2 = flights %>% 
        dplyr::select(year:day, hour, origin, dest, tailnum, carrier))
# # A tibble: 336,776 x 8
# year month   day  hour origin  dest tailnum carrier
# <int> <int> <int> <dbl>  <chr> <chr>   <chr>   <chr>
#         1  2013     1     1     5    EWR   IAH  N14228      UA
# 2  2013     1     1     5    LGA   IAH  N24211      UA
# 3  2013     1     1     5    JFK   MIA  N619AA      AA
# 4  2013     1     1     5    JFK   BQN  N804JB      B6
# 5  2013     1     1     6    LGA   ATL  N668DN      DL
# 6  2013     1     1     5    EWR   ORD  N39463      UA
# 7  2013     1     1     6    EWR   FLL  N516JB      B6
# 8  2013     1     1     6    LGA   IAD  N829AS      EV
# 9  2013     1     1     6    JFK   MCO  N593JB      B6
# 10  2013     1     1     6    LGA   ORD  N3ALAA      AA
# # ... with 336,766 more rows

# we want to add the full airline name to the flights2 data
# let's use left_join()
# the additional variable name will contain the full name
flights2 %>% 
        dplyr::select(-origin, -dest) %>% 
        left_join(airlines, by = "carrier")
# # A tibble: 336,776 x 7
# year month   day  hour tailnum carrier                     name
# <int> <int> <int> <dbl>   <chr>   <chr>                    <chr>
#         1  2013     1     1     5  N14228      UA    United Air Lines Inc.
# 2  2013     1     1     5  N24211      UA    United Air Lines Inc.
# 3  2013     1     1     5  N619AA      AA   American Airlines Inc.
# 4  2013     1     1     5  N804JB      B6          JetBlue Airways
# 5  2013     1     1     6  N668DN      DL     Delta Air Lines Inc.
# 6  2013     1     1     5  N39463      UA    United Air Lines Inc.
# 7  2013     1     1     6  N516JB      B6          JetBlue Airways
# 8  2013     1     1     6  N829AS      EV ExpressJet Airlines Inc.
# 9  2013     1     1     6  N593JB      B6          JetBlue Airways
# 10  2013     1     1     6  N3ALAA      AA   American Airlines Inc.
# # ... with 336,766 more rows


# you could have mutated this results == why we call them mutating joins
flights2 %>% 
        dplyr::select(-origin, -dest) %>% 
        mutate(name = airlines$name[match(carrier, airlines$carrier)])
# # A tibble: 336,776 x 7
# year month   day  hour tailnum carrier                     name
# <int> <int> <int> <dbl>   <chr>   <chr>                    <chr>
#         1  2013     1     1     5  N14228      UA    United Air Lines Inc.
# 2  2013     1     1     5  N24211      UA    United Air Lines Inc.
# 3  2013     1     1     5  N619AA      AA   American Airlines Inc.
# 4  2013     1     1     5  N804JB      B6          JetBlue Airways
# 5  2013     1     1     6  N668DN      DL     Delta Air Lines Inc.
# 6  2013     1     1     5  N39463      UA    United Air Lines Inc.
# 7  2013     1     1     6  N516JB      B6          JetBlue Airways
# 8  2013     1     1     6  N829AS      EV ExpressJet Airlines Inc.
# 9  2013     1     1     6  N593JB      B6          JetBlue Airways
# 10  2013     1     1     6  N3ALAA      AA   American Airlines Inc.
# # ... with 336,766 more rows


## Understanding Joins
# visual representation
# the key is used to match observations between the two tables
# variables in the second table will be "carried along for the ride"
# the number of matches on keys will provide the number of rows in the joined table 
x = tribble(
        ~key, ~val_x,
        1, "x1",
        2, "x2",
        3, "x3"
)

y = tribble(
        ~key, ~val_y,
        1, "y1",
        2, "y2",
        3, "y3"
)


## Inner Join
# simplest type of join is the inner join
# matches pairs of observations whenever thier keys are equal
# the output of an inner join is a new data frame that contains the key, x values and y values of the matches
# we need to specify for dplyr what the key to join by is = by
# UNMATCHED ROWS ARE NOT INCLUDED IN THE FINAL TABLE!!
x %>% inner_join(y, by = "key")
# # A tibble: 3 x 3
# key val_x val_y
# <dbl> <chr> <chr>
#         1     1    x1    y1
# 2     2    x2    y2
# 3     3    x3    y3

## Outer Join
# outer join keeps observations that appear in at least one of the tables
        # left join: keeps all observations in X
        # right join: keeps all observations in Y
        # full join: keeps all observations in X and Y
# this works by adding a "virtual" observation to each table -
# the unmatched observation will have a key that will always match and a NA value
# the left join should be your default join
# Venn Diagrams can't show what happens when key's don't uniquely identify an observation

## Duplicate Keys
# keys will not always be unique

# one table with duplicate keys: typical one-to-many relationship
# values that match the repeated keys will be placed once per matched row
x <- tribble(
        ~key, ~val_x,
        1, "x1",
        2, "x2",
        2, "x3",
        1, "x4"
)

y <- tribble(
        ~key, ~val_y,
        1, "y1",
        2, "y2"
)

left_join(x, y, by = "key")

# # A tibble: 4 x 3
# key val_x val_y
# <dbl> <chr> <chr>
#         1     1    x1    y1
# 2     2    x2    y2
# 3     2    x3    y2
# 4     1    x4    y1



# both tables have duplicate keys
# when you join with multiple keys we get all possible combinations, cartesian product
# this is usually an error if your key table has mutliple keys
# your data will not be a unique identifier per row
x <- tribble(
        ~key, ~val_x,
        1, "x1",
        2,"x2",
        2, "x3",
        3, "x4"
)

y <- tribble(
        ~key, ~val_y,
        1, "y1",
        2, "y2",
        2, "y3",
        3, "y4"
)

left_join(x, y, by = "key")
# # A tibble: 6 x 3
# key val_x val_y
# <dbl> <chr> <chr>
#         1     1    x1    y1
# 2     2    x2    y2
# 3     2    x2    y3
# 4     2    x3    y2
# 5     2    x3    y3
# 6     3    x4    y4


# defining the key columns
# the previous tables have all had a shared "key" variable
# this is not always the case - you can use other variables to join tables together
        # by = NULL = uses all variables that appear in both tables "natural join"

#natural join
flights2 %>% 
        left_join(weather) 
# Joining, by = c("year", "month", "day", "hour", "origin")
# # A tibble: 336,776 x 18
# year month   day  hour origin  dest tailnum carrier  temp  dewp humid
# <dbl> <dbl> <int> <dbl>  <chr> <chr>   <chr>   <chr> <dbl> <dbl> <dbl>
#         1  2013     1     1     5    EWR   IAH  N14228      UA    NA    NA    NA
# 2  2013     1     1     5    LGA   IAH  N24211      UA    NA    NA    NA
# 3  2013     1     1     5    JFK   MIA  N619AA      AA    NA    NA    NA
# 4  2013     1     1     5    JFK   BQN  N804JB      B6    NA    NA    NA
# 5  2013     1     1     6    LGA   ATL  N668DN      DL 39.92 26.06 57.33
# 6  2013     1     1     5    EWR   ORD  N39463      UA    NA    NA    NA
# 7  2013     1     1     6    EWR   FLL  N516JB      B6 39.02 26.06 59.37
# 8  2013     1     1     6    LGA   IAD  N829AS      EV 39.92 26.06 57.33
# 9  2013     1     1     6    JFK   MCO  N593JB      B6 39.02 26.06 59.37
# 10  2013     1     1     6    LGA   ORD  N3ALAA      AA 39.92 26.06 57.33
# # ... with 336,766 more rows, and 7 more variables: wind_dir <dbl>,
# #   wind_speed <dbl>, wind_gust <dbl>, precip <dbl>, pressure <dbl>,
# #   visib <dbl>, time_hour <dttm>


# join by a character vector 
# by = "x"
# this is similiar to a natural join but only uses some of the common variables
# for example flights and planes have the year variable but they mean different things
# we only want to join by tailnum
# common year variables will be noted with a suffix from which table they came from (.x or .y)
flights2 %>% 
        left_join(planes, by = "tailnum")
# # A tibble: 336,776 x 16
# year.x month   day  hour origin  dest tailnum carrier year.y
# <int> <int> <int> <dbl>  <chr> <chr>   <chr>   <chr>  <int>
#         1   2013     1     1     5    EWR   IAH  N14228      UA   1999
# 2   2013     1     1     5    LGA   IAH  N24211      UA   1998
# 3   2013     1     1     5    JFK   MIA  N619AA      AA   1990
# 4   2013     1     1     5    JFK   BQN  N804JB      B6   2012
# 5   2013     1     1     6    LGA   ATL  N668DN      DL   1991
# 6   2013     1     1     5    EWR   ORD  N39463      UA   2012
# 7   2013     1     1     6    EWR   FLL  N516JB      B6   2000
# 8   2013     1     1     6    LGA   IAD  N829AS      EV   1998
# 9   2013     1     1     6    JFK   MCO  N593JB      B6   2004
# 10   2013     1     1     6    LGA   ORD  N3ALAA      AA     NA
# # ... with 336,766 more rows, and 7 more variables: type <chr>,
# #   manufacturer <chr>, model <chr>, engines <int>, seats <int>,
# #   speed <int>, engine <chr>


# join by a named character vector
# by = c("a" = "b")
# this will match variable a in table x to variable b in table y
# the variables from x will be used in the output
flights2 %>% 
        left_join(airports, by = c("dest" = "faa"))
# # A tibble: 336,776 x 15
# year month   day  hour origin  dest tailnum carrier                            name      lat       lon   alt    tz   dst            tzone
# <int> <int> <int> <dbl>  <chr> <chr>   <chr>   <chr>                           <chr>    <dbl>     <dbl> <int> <dbl> <chr>            <chr>
#         1  2013     1     1     5    EWR   IAH  N14228      UA    George Bush Intercontinental 29.98443 -95.34144    97    -6     A  America/Chicago
# 2  2013     1     1     5    LGA   IAH  N24211      UA    George Bush Intercontinental 29.98443 -95.34144    97    -6     A  America/Chicago
# 3  2013     1     1     5    JFK   MIA  N619AA      AA                      Miami Intl 25.79325 -80.29056     8    -5     A America/New_York
# 4  2013     1     1     5    JFK   BQN  N804JB      B6                            <NA>       NA        NA    NA    NA  <NA>             <NA>
#         5  2013     1     1     6    LGA   ATL  N668DN      DL Hartsfield Jackson Atlanta Intl 33.63672 -84.42807  1026    -5     A America/New_York
# 6  2013     1     1     5    EWR   ORD  N39463      UA              Chicago Ohare Intl 41.97860 -87.90484   668    -6     A  America/Chicago
# 7  2013     1     1     6    EWR   FLL  N516JB      B6  Fort Lauderdale Hollywood Intl 26.07258 -80.15275     9    -5     A America/New_York
# 8  2013     1     1     6    LGA   IAD  N829AS      EV          Washington Dulles Intl 38.94453 -77.45581   313    -5     A America/New_York
# 9  2013     1     1     6    JFK   MCO  N593JB      B6                    Orlando Intl 28.42939 -81.30899    96    -5     A America/New_York
# 10  2013     1     1     6    LGA   ORD  N3ALAA      AA              Chicago Ohare Intl 41.97860 -87.90484   668    -6     A  America/Chicago
# # ... with 336,766 more rows


flights2 %>% 
        left_join(airports, c("origin" = "faa"))
# # A tibble: 336,776 x 15
# year month   day  hour origin  dest tailnum carrier                name      lat       lon   alt    tz   dst            tzone
# <int> <int> <int> <dbl>  <chr> <chr>   <chr>   <chr>               <chr>    <dbl>     <dbl> <int> <dbl> <chr>            <chr>
#         1  2013     1     1     5    EWR   IAH  N14228      UA Newark Liberty Intl 40.69250 -74.16867    18    -5     A America/New_York
# 2  2013     1     1     5    LGA   IAH  N24211      UA          La Guardia 40.77725 -73.87261    22    -5     A America/New_York
# 3  2013     1     1     5    JFK   MIA  N619AA      AA John F Kennedy Intl 40.63975 -73.77893    13    -5     A America/New_York
# 4  2013     1     1     5    JFK   BQN  N804JB      B6 John F Kennedy Intl 40.63975 -73.77893    13    -5     A America/New_York
# 5  2013     1     1     6    LGA   ATL  N668DN      DL          La Guardia 40.77725 -73.87261    22    -5     A America/New_York
# 6  2013     1     1     5    EWR   ORD  N39463      UA Newark Liberty Intl 40.69250 -74.16867    18    -5     A America/New_York
# 7  2013     1     1     6    EWR   FLL  N516JB      B6 Newark Liberty Intl 40.69250 -74.16867    18    -5     A America/New_York
# 8  2013     1     1     6    LGA   IAD  N829AS      EV          La Guardia 40.77725 -73.87261    22    -5     A America/New_York
# 9  2013     1     1     6    JFK   MCO  N593JB      B6 John F Kennedy Intl 40.63975 -73.77893    13    -5     A America/New_York
# 10  2013     1     1     6    LGA   ORD  N3ALAA      AA          La Guardia 40.77725 -73.87261    22    -5     A America/New_York
# # ... with 336,766 more rows

install.packages("maps")
library(maps)

# drawing a map of the US
airports %>% 
        semi_join(flights, c("faa" = "dest")) %>% 
        ggplot(aes(lon, lat)) +
        borders("state") +
        geom_point() +
        coord_quickmap()

# other implementations
# base::merge() can perform all four types of mutating joins
        # dplyr::inner_join(x,y) = merge(x,y)
        # dplyr::left_join(x,y) = merge(x,y, all.x = T)
        # dplyr::right_join(x,y) = merge(x,y, all.y = T)
        # dplyr::full_join(x,y) = merge(x,y, all.x = T, all.y = T)
# dplyr verbs more clearly convey the intent of your code: almost "hidden" in base R merge
# dplyr joins are faster and don't mess with the order of the rows

# SQL is the inspiration for dplyr's conventions - joins translate to SQL easily
        # dplyr::inner_join(x,y, by = "z") = SELECT * FROM x INNER JOIN y USING (z)
        # dplyr::left_join(x, y, by = "z") = SELECT * FROM x LEFT OUTER JOIN y USING (z)
        # dplyr::right_join(x, y, by = "z") = SELECT * FROM x RIGHT OUTER JOIN y USING (z)
        # dplyr::full_join(x, y, by = "z") = SELECT * FROM x FULL OUTER JOIN

# note that "INNER" and "OUTER" are optional

# joining on exact columns in SQL is slightly different notation
        # dplyr::inner_join(x,y, by = c("a" = "b")) = SELECT * FROM x INNER JOIN y ON x.a = y.b
# SQL supports a wider range of join types because you can connect the tables using constrains other than equality (non-equijoins)




## Filtering Joins
# filtering joins match observations in the same way as a mutating joins, but affect the observations, not the variables
        # semi_join(x,y) keeps all observations in x that have a match in y
        # anti_join(x,y) drops all observations in x that have a match in y

# semi-joins are useful for matching filtered summary tables back to the original rows
# imagine you've found the top 10 most popular destinations
top_dest <- flights %>% 
        count(dest, sort = T) %>% 
        head(10)

# # A tibble: 10 x 2
# dest     n
# <chr> <int>
#         1   ORD 17283
# 2   ATL 17215
# 3   LAX 16174
# 4   BOS 15508
# 5   MCO 14082
# 6   CLT 14064
# 7   SFO 13331
# 8   FLL 12055
# 9   MIA 11728
# 10   DCA  9705

# now you want to find each flight that went to one of these destinations
# you could filter yourself but it is difficult to extend this approach to multiple variables
flights %>% filter(dest %in% top_dest$dest)
# # A tibble: 141,145 x 19
# year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time arr_delay carrier
# <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>     <dbl>   <chr>
#         1  2013     1     1      542            540         2      923            850        33      AA
# 2  2013     1     1      554            600        -6      812            837       -25      DL
# 3  2013     1     1      554            558        -4      740            728        12      UA
# 4  2013     1     1      555            600        -5      913            854        19      B6
# 5  2013     1     1      557            600        -3      838            846        -8      B6
# 6  2013     1     1      558            600        -2      753            745         8      AA
# 7  2013     1     1      558            600        -2      924            917         7      UA
# 8  2013     1     1      558            600        -2      923            937       -14      UA
# 9  2013     1     1      559            559         0      702            706        -4      B6
# 10  2013     1     1      600            600         0      851            858        -7      B6
# # ... with 141,135 more rows, and 9 more variables: flight <int>, tailnum <chr>, origin <chr>,
# #   dest <chr>, air_time <dbl>, distance <dbl>, hour <dbl>, minute <dbl>, time_hour <dttm>


# instead you can use a semi-join, which connects the two tables like a mutating join
# but instead of adding new columns, semi_join only keeps the rows in x that have a match in y
flights %>% semi_join(top_dest)
# # A tibble: 141,145 x 19
# year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time arr_delay carrier
# <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>     <dbl>   <chr>
#         1  2013     1     1      542            540         2      923            850        33      AA
# 2  2013     1     1      554            600        -6      812            837       -25      DL
# 3  2013     1     1      554            558        -4      740            728        12      UA
# 4  2013     1     1      555            600        -5      913            854        19      B6
# 5  2013     1     1      557            600        -3      838            846        -8      B6
# 6  2013     1     1      558            600        -2      753            745         8      AA
# 7  2013     1     1      558            600        -2      924            917         7      UA
# 8  2013     1     1      558            600        -2      923            937       -14      UA
# 9  2013     1     1      559            559         0      702            706        -4      B6
# 10  2013     1     1      600            600         0      851            858        -7      B6
# # ... with 141,135 more rows, and 9 more variables: flight <int>, tailnum <chr>, origin <chr>,
# #   dest <chr>, air_time <dbl>, distance <dbl>, hour <dbl>, minute <dbl>, time_hour <dttm>

# only the existence of a match is important: it doesn't matter which observation is matched
# this means filtering joins never duplicate rows like mutating joins
# the inverse of a semi-join is an anti-join: an anti-join keeps the rows that don't have a match!
# anti_joins are useful for diagnosing join mismatches
# how many x don't have a match in y
flights %>% 
        anti_join(planes, by = "tailnum") %>% 
        count(tailnum, sort = T)
# # A tibble: 722 x 2
# tailnum     n
# <chr> <int>
#         1    <NA>  2512
# 2  N725MQ   575
# 3  N722MQ   513
# 4  N723MQ   507
# 5  N713MQ   483
# 6  N735MQ   396
# 7  N0EGMQ   371
# 8  N534MQ   364
# 9  N542MQ   363
# 10  N531MQ   349
# # ... with 712 more rows


## Join problems
# data may not always be clean...
        # start by identifying the variables from the primary key in each table
        # think about what the variables mean that you are connecting
        # you may find a key by guessing but this relationship may not always hold true
        # check that none of the variables in the primary key are missing - if a value if missing then it can't identify an observation!
        # check that your foriegn keys match primary keys in another table - do this with anti_join
        # checking the number of rows at the end of the join may not always ensure the join has worked - BE CAREFUL!


## Set Operations
# the final type of two-table verb are set operations
# all these operations work with a complete row, comparing the values of every variable
# these expect the x and y inputs to have the same variables: they treat observations like sets
        # intersect(x,y) = returns only observations in both x and y
        # union(x,y) = return unique observations in x and y
        # setdiff(x,y) = return observations in x, but not in y
df1 <- tribble(
        ~x, ~y,
        1,1,
        2,1
)

df2 <- tribble(
        ~x, ~y,
        1,1,
        1,2
)

# set operations
intersect(df1, df2)
# # A tibble: 1 x 2
# x     y
# <dbl> <dbl>
#         1     1     1

union(df1,df2)
# # A tibble: 3 x 2
# x     y
# <dbl> <dbl>
#         1     1     2
# 2     2     1
# 3     1     1

setdiff(df1, df2)
# # A tibble: 1 x 2
# x     y
# <dbl> <dbl>
#         1     2     1

setdiff(df2, df1)
# # A tibble: 1 x 2
# x     y
# <dbl> <dbl>
#         1     1     2



# chapter 11 strings ------------------------------------------------------



## Strings with stringr
# learn the basics of string manipulation in R
# focus of this chapter will be regular expressions
# regexp help manipulate patterns in strings
# work on structured and unstructured strings
library(stringr)

# basics
# you can create strings with either single quotes or double quites
string1 = "this is a string"
string2 = 'to put a "quote" inside a string, use single quotes'

# to include literal single or double quotes in a string you can use \ to "escape" it
double_quote = "\""
single_quote = '\''

# if you want to include a backslash in quotes you need to double the backslashes
# printed versions of the string will not show escapes: the raw text ouput is different than the printed output
# use writeLines() to see the raw text
x = c("\"","\\")
x
# [1] "\"" "\\"
writeLines(x)
# "
# \


# other special characters:
x = "\u00b5"
x
# [1] "µ"

# multiple strings can be stored in a character vector
char = c("one","two","three")




## string length
# base R has many functions that work with strings but stringr are more consistent
# all stringr verbs start with str_

# string length = number of characters in a string
str_length(c("a", "R for data science", NA))
# [1]  1 18 NA


# combining strings:
# use str_c()
str_c("x","y")
# [1] "xy"

str_c("x","y","z")
# [1] "xyz"


# NA valus are are contagious: if you want to print them as NA use str_replace_na()
x = c("abc", NA)
str_c("|-",x,"-|")
# [1] "|-abc-|" NA 
str_c("|-",str_replace_na(x), "-|")
# [1] "|-abc-|" "|-NA-|" 


# str_c is vectorized, it automatically recycles shorter vectors to the lame length as the longest:
str_c("prefix-",c("a","b","c"), "-suffix")
# [1] "prefix-a-suffix" "prefix-b-suffix" "prefix-c-suffix"

# objects of length 0 are silently dropped...
name = "hadley"
time_of_day = "morning"
birthday = F

str_c(
        "Good", time_of_day, " ", name,
        if (birthday) " and HAPPY BIRTHDAY",
        "."
)

# [1] "Goodmorning hadley."

# to collapse a vector of strings into a single string use collapse
str_c(c("x","y","z"), collapse = ", ")
# [1] "x, y, z"





## subsetting strings
# you can extract parts of strings using str_sub()
# feed start and end arguements that give the inclusive position of the substring
x = c("Apple", "Bananna", "Pear")
str_sub(x, 1, 3)
# [1] "App" "Ban" "Pea"

# negative numbers count backwards from the end of a string
str_sub(x,-3, -1)
# [1] "ple" "nna" "ear"

# note that str_sub won't fail if the string is too short: it will just return as much as possible
str_sub("a",1,5)
# [1] "a"


# you can also use the assignment form to modify strings
str_sub(x,1,1) = str_to_lower(str_sub(x,1,1))
x
# [1] "apple"   "bananna" "pear" 


## Locales
# changing case is based on your locale: check when using str_to_lower or str_to_upper
# you can also use str_to_title
str_to_upper(c("i","I"))
# [1] "I" "I"

# setting turkish locale
str_to_upper(c("i","I"), locale = "tr")


# sorting is also affected by locale
# base R sort and order sort strings based on the current locale
# use str_sort and str_order which can take a locale arguement
x = c("apple","eggplant","banana")

str_sort(x, locale = "en")
# [1] "apple"    "banana"   "eggplant"
str_sort(x, locale = "haw")
# [1] "apple"    "eggplant" "banana"








## Matching Patterns with Regular Expressions
# regex let you describe patterns in strings
# str_view and str_view_all take a character vector and a regex and show you how they match
# the regex expression will eventually be embedded in the str_verbs()

# basic matches
# matching exact strings
x = c("apple","banana","pear")
str_view(x, "an")

# matching character within a string
str_view(x, ".a.")

# how to escape characters - matching on "."
# use the backslash to escape special behavior
# regex("\.") will escape period...but we also need to escape the backslash!
# the actual expression will only contain one backslash
dot = "\\."
writeLines(dot)
# \.
str_view(c("abc","a.c","bef"), "a\\.c")


# matching a backslash - you actually need four backslashes to match the literal 1 backslash
x = "a\\b"
writeLines(x)
str_view(x, "\\\\")
# a\b


# Regular Expression Anchors
# by default regular expressions will match any part of a string
# you can anchor an expression so that it matches from the start or end of a string
        # ^ to match the start of a string
        # $ to match the end of the string
# if you begin with power (^) you end with money ($)
x = c("apple","banana","pear")
str_view(x, "^a")

# force a regular expression to only match a complete string
# anchor it with both power and money (^ and $)
x = c("apple pie", "apple", "apple cake")

# finds all instances of apple in any string
str_view(x, "apple")

# finds only the word "apple" without any other strings
str_view(x, "^apple$")


## Character Classes and Alternatives
# other special characters...
        # \d matches any digit
        # \s any whitespace
        # [abc] matches a, b, or c
        # [^abc] matches anything except a, b, or c
# remember to create regular expressions containing \ you'll need to escape the \ for the string
# you type \\d or \\s but the actual regular expression will be \d or \s

# you can use alternation to pick between one or more alternative patterns
# example: abc|d..f will match either "abc" or "deaf"
str_view(c("grey","gray"), "gr(e|a)y")


# Repetition
# control how many times a pattern matches:
        # ?: 0 or 1
        # +: 1 or more
        # *: 0 or more
x = "1888 is the longest year in Roman Numerals: MDCCCLXXXVIII"
# matches the first instance of "CC"
str_view(x, "CC?")
# matches 1 or more instances of "CC"
str_view(x, "CC+")
# matches 1 or more of L or X after C
str_view(x, "C[LX]+")


# you can also specify the number of matches exactly
        # {n}: exactly n matches
        # {n,}: n or more matches
        # {,m}: at most m matches
        # {n,m}: between n and m matches
# by default these matches are "greedy" and will match the longest string possible

# exactly two matches of C
str_view(x, "C{2}")

# two or more matches of C
str_view(x, "C{2,}")

# matches of C between 2 and 3
str_view(x, "C{2,3}")


# to match the shortest string possible = "lazy" = put a question after them
str_view(x, "C{2,3}?")
str_view(x, "C[LX]+?")


# Grouping and Backreferences
# parentheses as a way to group complex expressions
# these groups  that you can refer to with backreferences like \1, \2
# find all fruits that have a repeated pair of letters
str_view(fruit,"(..)\\1", match = T)




## Tools
# now that we know the basics of regular expressions let's apply them
# we can use stringr functions to apply our regex knowledge
        # determine which strings match a pattern
        # find the positions of matches
        # extract the content of matches
        # replace matches with new variables
        # split a string based on a match
# good idea to breakup expressions into smaller regexp peices for large problems

## Detect Matches
# to determine if a character vector matches a pattern, use str_detect()
# this returns a logical vector the same length as the input
# what strings match "e" in the list of fruits?
x = c("apple", "banana", "pear")
str_detect(x, "e")
# [1]  TRUE FALSE  TRUE


# remember when you use a logical vector in a numeric context = FALSE = 0 and TRUE = 1
# we can perform match on these operations

# sum how many words start with t
sum(str_detect(words,"^t"))
# [1] 65

# what proportion of common words end with a vowel?
mean(str_detect(words,"[aeiou]$"))
# [1] 0.2765306ß


# stringing together multiple expressions
# find all words that don't contain any vowels:

# find all words containing at least one vowel and negate
no_vowels_1 = !str_detect(words, "[aeiou]")

# find all words consisting only of consonants (non-vowels)
no_vowels_2 = str_detect(words, "^[^aeiou]+$")

identical(no_vowels_1, no_vowels_2)
# [1] TRUE


# select the elements that match a pattern
# to do this use str_subset()

# logical subsetting solution - words that end in x
words[str_detect(words, "x$")]
# [1] "box" "sex" "six" "tax"

# str_subset solution - words that end in x
str_subset(words, "x$")
# [1] "box" "sex" "six" "tax"


# you can combine filter and str_subset to filter a data frame column

# create the words data frame for example
df = tibble(
        word = words,
        i = seq_along(word)
)

# filter the words data frame by a subset of the matched regular expression
# filter the words of the data frame that end in x
df %>% filter(str_detect(words, "x$"))
# # A tibble: 4 x 2
# word      i
# <chr> <int>
#         1 box     108
# 2 sex     747
# 3 six     772
# 4 tax     841



# a variation of str_detect is str_count:
# rather than just a yes or no, str_count will tell you how many matches there are per string!!
x = c("apple", "banana", "pear")
str_count(x, "a")
# [1] 1 3 1

# on average how many vowels per word?
mean(str_count(words, "[aeiou]"))
# [1] 1.991837


# combining str_count with mutate
df %>% 
        mutate(
                vowels = str_count(word, "[aeiou]"),
                consonants = str_count(word, "[^aeiou]")
        )

# # A tibble: 980 x 4
# word         i vowels consonants
# <chr>    <int>  <int>      <int>
#         1 a            1      1          0
# 2 able         2      2          2
# 3 about        3      3          2
# 4 absolute     4      4          4
# 5 accept       5      2          4
# 6 account      6      3          4
# 7 achieve      7      4          3
# 8 across       8      2          4
# 9 act          9      1          2
# 10 active      10      3          3
# # ... with 970 more rows


# note that matches never overlap
# example: "abababa" how many times will the pattern "aba" match? Regexp say TWO not THREE
str_count("abababa","aba")
# [1] 2
str_view_all("abababa","aba")

# note the use of str_view_all!
# many stringr functions come in pairs:
        # one function with a single match
        # _all function with all matches


## Exact Matches
# to extract the actual text of a match use str_extract
stringr::sentences
length(sentences);head(sentences)

# find matches that contain a color
# create vector of color names than match using regular expression
colors <- c(
        "red", "orange", "yellow", "green", "blue", "purple"
)

# create regex with all our colors
color.match <- str_c(colors, collapse = "|")
color.match

# select sentences with a color then extract the color 
has_color <- str_subset(sentences, color.match)
matches <- str_extract(has_color, color.match)
head(matches)
# [1] "blue" "blue" "red"  "red"  "red"  "blue"

# note that str_extract only gives us the first match
# some sentences have more than one match
more <- sentences[str_count(sentences, color.match) > 1]
str_view_all(more, color.match)

# only first match return
str_extract(more, color.match)
# [1] "blue"   "green"  "orange"

# to get all matches use the partner of str_extract, str_extract_all
str_extract_all(more, color.match)
# [[1]]
# [1] "blue" "red" 
# 
# [[2]]
# [1] "green" "red"  
# 
# [[3]]
# [1] "orange" "red" 

# if we use the arguement simplify = T to return a matrix
# matrix will be expanded to be as long as the number of the most matches
str_extract_all(more, color.match, simplify = T)
# [,1]     [,2] 
# [1,] "blue"   "red"
# [2,] "green"  "red"
# [3,] "orange" "red"

x <- c("a", "a b", "a b c")
str_extract_all(x,"[a-z]", simplify = T)
# [,1] [,2] [,3]
# [1,] "a"  ""   ""  
# [2,] "a"  "b"  ""  
# [3,] "a"  "b"  "c" 


## Grouped Matches
# we can use paraenthsis to extract parts of a complicated match
# imagine we want to extract nouns from sentences - any word that comes after "a" or "the"
# we define a word of at least one character that is not a space
noun <- "(a|the) ([^ ]+)"

has_noun <- sentences %>% 
        str_subset(noun) %>% 
        head(25)

has_noun %>% str_extract(noun)
# [1] "the smooth"  "the sheet"   "the depth"   "a chicken"   "the parked"  "the sun"     "the huge"    "the ball"   
# [9] "the woman"   "a helps"     "the man's"   "the sea."    "the booth"   "a hole"      "the bent"    "the pants"  
# [17] "the view"    "the tank."   "the tall"    "the same"    "the load"    "the winding" "the size"    "the grease" 
# [25] "the coat" 

# str_extract gives us the complete match
# str_match gives each individual component in a matrix
# one column for the complete match and additional columns for each group
has_noun %>% str_match(noun)
# [,1]          [,2]  [,3]     
# [1,] "the smooth"  "the" "smooth" 
# [2,] "the sheet"   "the" "sheet"  
# [3,] "the depth"   "the" "depth"  
# [4,] "a chicken"   "a"   "chicken"
# [5,] "the parked"  "the" "parked" 
# [6,] "the sun"     "the" "sun"    
# [7,] "the huge"    "the" "huge"   
# [8,] "the ball"    "the" "ball"   
# [9,] "the woman"   "the" "woman"  
# [10,] "a helps"     "a"   "helps"  
# [11,] "the man's"   "the" "man's"  
# [12,] "the sea."    "the" "sea."   
# [13,] "the booth"   "the" "booth"  
# [14,] "a hole"      "a"   "hole"   
# [15,] "the bent"    "the" "bent"   
# [16,] "the pants"   "the" "pants"  
# [17,] "the view"    "the" "view"   
# [18,] "the tank."   "the" "tank."  
# [19,] "the tall"    "the" "tall"   
# [20,] "the same"    "the" "same"   
# [21,] "the load"    "the" "load"   
# [22,] "the winding" "the" "winding"
# [23,] "the size"    "the" "size"   
# [24,] "the grease"  "the" "grease" 
# [25,] "the coat"    "the" "coat"


# if your data is a tibble you can use tidyr::extract
# this works like str_match but requires you to name your matches
# matches are placed in new columns
tibble(sentence = sentences) %>% 
        tidyr::extract(
                sentence, c("article", "noun"), "(a|the) ([^ ]+)",
                remove = F
        )
# # A tibble: 720 x 3
# sentence                                    article noun   
# * <chr>                                       <chr>   <chr>  
#         1 The birch canoe slid on the smooth planks.  the     smooth 
# 2 Glue the sheet to the dark blue background. the     sheet  
# 3 It's easy to tell the depth of a well.      the     depth  
# 4 These days a chicken leg is a rare dish.    a       chicken
# 5 Rice is often served in round bowls.        NA      NA     
# 6 The juice of lemons makes fine punch.       NA      NA     
# 7 The box was thrown beside the parked truck. the     parked 
# 8 The hogs were fed chopped corn and garbage. NA      NA     
# 9 Four hours of steady work faced us.         NA      NA     
# 10 Large size in stockings is hard to sell.    NA      NA     
# # ... with 710 more rows


# like str_extract we can use str_match_all to give all matches for each string



## Replacing Matches
# str_replace and str_replace_all allow you to replace matches of strings
# simple is to replace a pattern with a fixed string
x <- c("apple", "pear", "banana")
str_replace(x, "[aeiou]", "-")
# [1] "-pple"  "p-ar"   "b-nana"

str_replace_all(x, "[aeiou]", "-")
# [1] "-ppl-"  "p--r"   "b-n-n-"

# with str_replace_all you can perform multiple replacements by supplying a named vector
x <- c("1 house", "2 cars", "3 people")
str_replace_all(x, c("1" = "one", "2" = "two", "3" = "three"))
# [1] "one house"    "two cars"     "three people"

# instead of replacing a fixed string you can use backreferences to insert components of the match
# in the following example we flip the order of the second and third words:
sentences %>% 
        str_replace("([^ ]+) ([^ ]+) ([^ ]+)", "\\1 \\3 \\2") %>% 
        head(5)
# [1] "The canoe birch slid on the smooth planks."  "Glue sheet the to the dark blue background."
# [3] "It's to easy tell the depth of a well."      "These a days chicken leg is a rare dish."   
# [5] "Rice often is served in round bowls." 

## Splitting
# use str_split to split a string up into pieces
sentences %>% 
        head(5) %>% 
        str_split(" ")
# [[1]]
# [1] "The"     "birch"   "canoe"   "slid"    "on"      "the"     "smooth"  "planks."
# 
# [[2]]
# [1] "Glue"        "the"         "sheet"       "to"          "the"         "dark"        "blue"        "background."
# 
# [[3]]
# [1] "It's"  "easy"  "to"    "tell"  "the"   "depth" "of"    "a"     "well."
# 
# [[4]]
# [1] "These"   "days"    "a"       "chicken" "leg"     "is"      "a"       "rare"    "dish."  
# 
# [[5]]
# [1] "Rice"   "is"     "often"  "served" "in"     "round"  "bowls."

# becuase a split might contain different number of peices the result is a list
# if we want to extract elements we can extract elements from this list easily
"a|b|c|d" %>% 
        str_split("\\|") %>% 
        .[[1]]
# [1] "a" "b" "c" "d"

# other wise you can use simplify = T to have stringr return a matrix of all the splits
sentences %>% 
        head(5) %>% 
        str_split(" ", simplify = T)
# [,1]    [,2]    [,3]    [,4]      [,5]  [,6]    [,7]     [,8]          [,9]   
# [1,] "The"   "birch" "canoe" "slid"    "on"  "the"   "smooth" "planks."     ""     
# [2,] "Glue"  "the"   "sheet" "to"      "the" "dark"  "blue"   "background." ""     
# [3,] "It's"  "easy"  "to"    "tell"    "the" "depth" "of"     "a"           "well."
# [4,] "These" "days"  "a"     "chicken" "leg" "is"    "a"      "rare"        "dish."
# [5,] "Rice"  "is"    "often" "served"  "in"  "round" "bowls." ""            ""  


# we can also request a maximum number of peices
fields <- c("Name: Hadley", "Country: NZ", "Age: 35")
fields %>% 
        str_split(": ", n = 2, simplify = T)
# [,1]      [,2]    
# [1,] "Name"    "Hadley"
# [2,] "Country" "NZ"    
# [3,] "Age"     "35" 


# instead of splitting up strings by patterns...
# we can also split by character, line, sentence and word boundary()s
x <- "This is a sentence. This is another sentence."
str_view_all(x, boundary("word"))

str_split(x, " ")[[1]]
# [1] "This"      "is"        "a"         "sentence." "This"      "is"        "another"   "sentence."

str_split(x, boundary("word"))[[1]]
# [1] "This"     "is"       "a"        "sentence" "This"     "is"       "another"  "sentence"


# str_locate and str_locate_all give you the starting and ending points of each match
# use str_locate to find the matching pattern and str_sub to extract or modify the pattern



## Other Types of Pattern
# when you use a pattern that is a string its automatically wrapped into a call to regex()

# the regular call...
str_view(fruit, "nana")

# is shorthand for...
str_view(fruit, regex("nana"))


# we can use other arguements of the regex() function to control parameters of the match

# ignore_case allows characters to match either upper or lower case letters
bananas <- c("banana", "Banana", "BANANA")
str_view(bananas, "banana")
str_view(bananas, regex("banana", ignore_case = T))

# multiline allows ^ and $ to match the start and end of each line rather than of each string
x <- "Line 1\nLine 2\nLine 3"
str_extract_all(x, "^Line")[[1]]
# [1] "Line"
str_extract_all(x, regex("^Line", multiline = T))[[1]]
# [1] "Line" "Line" "Line"

# comments allows you to use commands and white space to make complex regex more understandable
# put your explainitory comments within the regex statement
phone <- regex("
               \\(? # optional opening parens
               (\\d{3}) # area code
               [)- ]? # optional closing parens, dash or space",
               comments = T
)

str_match("514-791-8141", phone)


# there are three other functions you can use instead of regex()
# fixed() matches exactly the specific sequence of bytes
# it ignores all other special regular expressions and operates at a very low level
# this allows you to avoid complex escaping and can be much faster than regular expressions
# fixed performs better than regex in some cases
# but fixed has problems with non-english data -fixed will have trouble with the multiple ways to represent certain characters
# coll() compares strings using standard collation rules - this is useful for doing case sensititve matching
# col takes a locale parameter that controls which rules are used for which characters
# both fixed() and regex() have ignore_case arguements but they do not allow you to pick the locale
# the downside of coll() is speed - regex and fixed both perform better
library(dplyr);library(stringr);library(stringi)

fixed = str_detect(sentences, fixed("the"))
head(fixed, 10)
# [1]  TRUE  TRUE  TRUE FALSE FALSE FALSE  TRUE FALSE FALSE FALSE

regex = str_detect(sentences, "the")
head(regex, 10)
# [1]  TRUE  TRUE  TRUE FALSE FALSE FALSE  TRUE FALSE FALSE FALSE

a1 <- "\u00e1"
a2 <- "a\u0301"
c(a1, a2)
# [1] "á" "á"
a1 == a2
# [1] FALSE


str_detect(a1, fixed(a2))
# [1] FALSE

str_detect(a1, coll(a2))
# [1] TRUE


# check your default locale
stringi::stri_locale_info()
# $Language
# [1] "en"
# 
# $Country
# [1] "US"
# 
# $Variant
# [1] ""
# 
# $Name
# [1] "en_US"

# you can use boundary() to match boundaries in other functions:
x <- "this is a sentence"
str_view_all(x, boundary("word"))

str_extract_all(x, boundary("word"))
# [[1]]
# [1] "this"     "is"       "a"        "sentence"

## other uses of Regular Expressions
# there are two useful functions in base R  that also use regular expressions
# apropos() = useful if you cannot remember the name of a function
# it will return all functions based on your regex statement within apropos
apropos("replace")
# [1] ".rs.registerReplaceHook"      ".rs.replaceBinding"           "replace"                     
# [4] "setReplaceMethod"             "str_replace"                  "str_replace_all"             
# [7] "str_replace_na"               "stri_replace"                 "stri_replace_all"            
# [10] "stri_replace_all_charclass"   "stri_replace_all_coll"        "stri_replace_all_fixed"      
# [13] "stri_replace_all_regex"       "stri_replace_first"           "stri_replace_first_charclass"
# [16] "stri_replace_first_coll"      "stri_replace_first_fixed"     "stri_replace_first_regex"    
# [19] "stri_replace_last"            "stri_replace_last_charclass"  "stri_replace_last_coll"      
# [22] "stri_replace_last_fixed"      "stri_replace_last_regex"      "stri_replace_na"



# dir() lists all files in a directory 
# the pattern arguement takes a regular expression and only returns filenames with a match
# example below provides all .rmd files in our directory 
head(dir(pattern = "\\.Rmd$"))
# [1] "Final Project_Practical Machine Learning_v2.Rmd" "Final Project_Practical Machine Learning.Rmd"   
# [3] "motor.trend.analysis.Rmd"                        "motor.trend.fuel.study.Rmd"                     
# [5] "Practical Machine Learning - Project.Rmd"        "Practical_Machine_Learning_-_Project.Rmd" 


## Stringi
# stringr is built on top of the stringi package
# stringr is useful when learning because you are limited to a minimal set of functions
# these minimal functions handle the most common string manipulation functions
# stringi is designed to be comprehensive - it contains almost every function you will ever need to string manipulation
# stringi has 234 functions to just 42 in stringr






# chapter 12 factors ------------------------------------------------------




## Chapter 12: Factors with forcasts
# in R factors are used to work with categorical variables
# variables that have fixed and known set of possible values
# historically factors were much easier to work with than character values!!
# many of the base functions convert characters to factors - and sometimes this is bad
# converting automatically to factors can be a big problem if you are not careful
# in the tidyverse you do not have to worry about these issues - you can focus on simulations were factors are needed for what they are intended for

# to work with factors we will use the forcats package
# forcats gives us tools to use FOR CATegorical variables
# it provides a wide range of helpers for working with factors
# forcats is not apart of core tidyverse so we need to load it explicitly
library(tidyverse)
library(forcats)

# creating factors
# imagine you have a string of variable months
x1 <- c("Dec", "Apr", "Jan", "Mar")

# using a string to record these months has a few problems:
# there are only twelve possible months and nothing is saving us from typos
x2 <- c("Dec", "Apr", "Jam", "Mar")

# these character months do not sort in a useful way - the sort is actually wrong
sort(x1)
# [1] "Apr" "Dec" "Jan" "Mar"

# we can fix both of these problems by using a factor
# to create a factor we must first build valid levels
month_levels <- c(
        "Jan", "Feb","Mar","Apr", "May", "Jun",
        "Jul","Aug", "Sep", "Oct", "Nov", "Dec" 
)

# using these defined levels we can now create a factor
y1 <- factor(x1, levels = month_levels)
y1
# [1] Dec Apr Jan Mar
# Levels: Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec

# sorting the variable now gives us the correct monthly sort
sort(y1)
# [1] Jan Mar Apr Dec
# Levels: Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec

# any values not in the factor levels will be treated as NA
y2 <- factor(x2, levels = month_levels)
y2
# [1] Dec  Apr  <NA> Mar 
# Levels: Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec

# if you want an error to return use parse_factor from readr
y3 <- parse_factor(x2, levels = month_levels)
# Warning: 1 parsing failure.

# if we omit the levels arguement they'll be defined "as is" from the data in alphabetical order
# in this case this is not correct for a monthly sort
factor(x1)
# [1] Dec Apr Jan Mar
# Levels: Apr Dec Jan Mar

# we might like that the order of the levels match the order of thier first appearance in the data
# we can do this when creating the factor by setting levels to unique() and with fct_inorder()
f1 <- factor(x1, levels = unique(x1))
f1
# [1] Dec Apr Jan Mar
# Levels: Dec Apr Jan Mar

f2 <- x1 %>% factor() %>% fct_inorder()
f2
# [1] Dec Apr Jan Mar
# Levels: Dec Apr Jan Mar


# we can access the defined levels easily 
levels(f2)
# [1] "Dec" "Apr" "Jan" "Mar"


## General Social Survey
# for the rest of this chapter we will focus on forcats::gss_cat
# this is a sample from the General Social Survey
# we subset the data to a few select columns that we use to illustrate challenges with factors

# view the gss data
head(gss_cat, 10)
# # A tibble: 10 x 9
# year marital         age race  rincome        partyid            relig              denom             tvhours
# <int> <fct>         <int> <fct> <fct>          <fct>              <fct>              <fct>               <int>
#         1  2000 Never married    26 White $8000 to 9999  Ind,near rep       Protestant         Southern baptist       12
# 2  2000 Divorced         48 White $8000 to 9999  Not str republican Protestant         Baptist-dk which       NA
# 3  2000 Widowed          67 White Not applicable Independent        Protestant         No denomination         2
# 4  2000 Never married    39 White Not applicable Ind,near rep       Orthodox-christian Not applicable          4
# 5  2000 Divorced         25 White Not applicable Not str democrat   None               Not applicable          1
# 6  2000 Married          25 White $20000 - 24999 Strong democrat    Protestant         Southern baptist       NA
# 7  2000 Never married    36 White $25000 or more Not str republican Christian          Not applicable          3
# 8  2000 Divorced         44 White $7000 to 7999  Ind,near dem       Protestant         Lutheran-mo synod      NA
# 9  2000 Married          44 White $25000 or more Not str democrat   Protestant         Other                   0
# 10  2000 Married          47 White $25000 or more Strong republican  Protestant         Southern baptist        3

# when factors are stored in a tibble - we can't see thier levels easily
# one way to see them is with count()
gss_cat %>% 
        count(race)
# # A tibble: 3 x 2
# race      n
# <fct> <int>
#         1 Other  1959
# 2 Black  3129
# 3 White 16395

# we can also see the factors available in a bar chart
ggplot(data = gss_cat, aes(race)) +
        geom_bar()

# be careful ggplot will drop levels that don't have any values automatically
# we can force ggplot to show the results with a drop = F statement
ggplot(gss_cat, aes(race)) +
        geom_bar() +
        scale_x_discrete(drop = F)

# these levels represent valid values that simply did not occur in this data set (i.e. Not Applicable)
# when you work with factors the two most common operations are:
        # changing the order of the factors
        # changing the values of the levels


## Modifying Factor Order
# we often want to change the order of the factor levels in a visualizaiton
# we want to plot the values as factors in the best way that we see fit for the data 

# plotting with factors example
relig <- gss_cat %>% 
        group_by(relig) %>% 
        summarize(
                age = mean(age, na.rm = T),
                tvhours = mean(tvhours, na.rm = T),
                n = n()
        )
ggplot(relig, aes(tvhours, relig)) + geom_point()


# we can't interpret this plot because of the factor order
# data points a split all over the place
# we can re-order the data by tvhours
# we can improve it by reordering the levels of relig using fct_reorder()
# fct_reorder takes three arguements:
        # f, the factor whose levels you want to reorder
        # x, a numeric vector that you want to use to reorder the levels
        # optionally a fun, or function that is useful if there are multiple values of x for each value of f
        # default fun is median

# our plot is now ordered in a logical way from religion with  highest tv hours to religion with  lowest
ggplot(relig, aes(tvhours, fct_reorder(relig, tvhours))) +
        geom_point()

# one best practice is to move the transformations out of the aes() step and into a seperate mutate() step
relig %>% 
        mutate(relig = fct_reorder(relig, tvhours)) %>% 
        ggplot(aes(tvhours, relig)) +
        geom_point()


# what if we were to create a similiar plot looking at average age across income level?
rincome <- gss_cat %>% 
        group_by(rincome) %>% 
        summarize(
                age = mean(age, na.rm = T),
                tvhours = mean(tvhours, na.rm = T),
                n = n()
        )

# plot the data
# here we should not have reordered our factors - they dataset already gave them in the correct order
ggplot(rincome, aes(age, fct_reorder(rincome, age))) +
        geom_point()

# however - what if we want to remove some factors that do not provide value based on data?
ggplot(rincome, aes(age, fct_relevel(rincome, "Not applicable"))) +
        geom_point()

# another type of reordering is useful wen you are coloring the lines on a plot
# fct_reorder2 reorders the factor by the y values assoicaited with the largest x values
# this makes the plot eaiser to read because the line colors line up with the legend
by_age <- gss_cat %>% 
        filter(!is.na(age)) %>% 
        group_by(age, marital) %>% 
        count() %>% 
        mutate(prop = n() / sum(n))

ggplot(by_age, aes(age, prop, color = marital)) +
        geom_line(na.rm = T)


# using fct_reorder2 to relevel the factors based on color scheme
ggplot(by_age, aes(age, prop, color = fct_reorder2(marital, age, prop))) +
        geom_line()+
        labs(color = "marital")

# refactoring with bar plots
# use fct_infreq() to order levels of factors by increasing frequency: this does not require a variable to sort by
# we can combine this with fct_rev() to reverse the order of our barplots
gss_cat %>% 
        mutate(marital = marital %>% fct_infreq() %>% fct_rev()) %>% 
        ggplot(aes(marital)) +
        geom_bar()

# this concludes the section or reordering or releveling factor values!!




## Modifying Factor Levels
# more powerful than just changing the orders of the levels of a factor - we can change thier values!!
# the most general case is fct_recode()
# this allows you to recode or change the value of each level in our factor

gss_cat %>% count(partyid)
# A tibble: 10 x 2
# partyid                n
# <fct>              <int>
#         1 No answer            154
# 2 Don't know             1
# 3 Other party          393
# 4 Strong republican   2314
# 5 Not str republican  3032
# 6 Ind,near rep        1791
# 7 Independent         4119
# 8 Ind,near dem        2499
# 9 Not str democrat    3690
# 10 Strong democrat     3490

# these levels are inconsistent - we can tweak and relevel these factors
gss_cat %>% 
        mutate(partyid = fct_recode(partyid,
                                    "Republican, strong" = "Strong Republician",
                                    "Republican, weak" = "Not str republican",
                                    "Independent, near rep" = "Ind, near rep",
                                    "Independent, near dem" = "Ind, near dem",
                                    "Democrat, weak" = "Not str democrat",
                                    "Democrat, strong" = "Strong democrat")) %>% 
        count(partyid)
# A tibble: 10 x 2
# partyid               n
# <fct>             <int>
#         1 No answer           154
# 2 Don't know            1
# 3 Other party         393
# 4 Strong republican  2314
# 5 Republican, weak   3032
# 6 Ind,near rep       1791
# 7 Independent        4119
# 8 Ind,near dem       2499
# 9 Democrat, weak     3690
# 10 Democrat, strong   3490

# fct_recode will leave levels that aren't mentioned as is
# it will warn you if you accidentally refer to a level that does not exist

# to combine groups - you can assign multiple old levels to the same new level
gss_cat %>% 
        mutate(partyid = fct_recode(partyid,
                                    "Republican, strong" = "Strong Republician",
                                    "Republican, weak" = "Not str republican",
                                    "Independent, near rep" = "Ind, near rep",
                                    "Independent, near dem" = "Ind, near dem",
                                    "Democrat, weak" = "Not str democrat",
                                    "Democrat, strong" = "Strong democrat",
                                    "Other" = "No answer",
                                    "Other" = "Don't know",
                                    "Other" = "Other party")) %>% 
        count(partyid)
# A tibble: 8 x 2
# partyid               n
# <fct>             <int>
#         1 Other               548
# 2 Strong republican  2314
# 3 Republican, weak   3032
# 4 Ind,near rep       1791
# 5 Independent        4119
# 6 Ind,near dem       2499
# 7 Democrat, weak     3690
# 8 Democrat, strong   3490        

# you must use this technique with care!!
# if you group together categories that are truly different you will end up with misleading results

# if you want to collapse a lot of levels you can use fct_collapse
# for each new variable you can provide a vector of old levels
gss_cat %>% 
        mutate(partyid = fct_collapse(partyid,
                                      other = c("No answer", "Don't know", "Other party"),
                                      rep = c("Strong republican", "Not str republican"),
                                      ind = c("Ind,near rep", "Independent", "Ind,near dem"),
                                      dem = c("Not str democrat", "Strong democrat"))) %>% 
        count(partyid)
# A tibble: 4 x 2
# partyid     n
# <fct>   <int>
#         1 other     548
# 2 rep      5346
# 3 ind      8409
# 4 dem      7180


# sometimes you want to lump together all the small groups to make a plot or table simpler
# try fct_lump()
gss_cat %>% 
        mutate(relig = fct_lump(relig)) %>% 
        count(relig)
# relig          n
# <fct>      <int>
#         1 Protestant 10846
# 2 Other      10637

# we can use a n parameter to specific how many groups excluding other we want to keep
gss_cat %>% 
        mutate(relig = fct_lump(relig, n = 10)) %>% 
        count(relig, sort = T) %>% 
        print(n = Inf)
# A tibble: 10 x 2
# relig                       n
# <fct>                   <int>
#         1 Protestant              10846
# 2 Catholic                 5124
# 3 None                     3523
# 4 Christian                 689
# 5 Other                     458
# 6 Jewish                    388
# 7 Buddhism                  147
# 8 Inter-nondenominational   109
# 9 Moslem/islam              104
# 10 Orthodox-christian         95











# chapter 13 dates and times ----------------------------------------------



## Chapter 13: Dates and Times with lubridate
# dates and times are hard because they need to reconcile two physical peices:
# earth rotating around the sun and all geogrpahy 
# this chapter will cover using lubridate to help with analysis dates and times
library(lubridate)
library(nycflights13)


# there are three types of date / time that refer to an instant in time
        # a date <date>
        # time within a day <time>
        # date-time is a date plus a time <dttm>
# R doesn't have a base package class for storing times
# always use the simplest data for your needs - don't add date time if you only need date

# get current date or date time
today()
# [1] "2018-03-02"
now()
# [1] "2018-03-02 11:56:07 PST"

# there are three ways we will likely create a date/time
        # from a string
        # from individual date time components
        # from an existing date/time object
# we will review each going forward

# Dates from Strings
# date data often comes as strings
# we can use lubridate to parse out these strings and convert them to actual date objects
# they automatically work out the format one you specifiy the order of the component
# to use them identify the order that year, month, and day appear in your dates
# then arrange your "format" option to match the string coming in to the lubridate statement

#year month day
ymd("2017-01-31")
# [1] "2017-01-31"

# month day year
mdy("January 31st, 2017")
# [1] "2017-01-31"

# day month year
dmy("31-Jan-2017")
# [1] "2017-01-31"

# these functions also take unquoted numbers
ymd(20170131)
# [1] "2017-01-31"


# create date time objects with lubridate
# specify additional hms to the date parsing function
ymd_hms("2017-01-31 20:11:59")
# [1] "2017-01-31 20:11:59 UTC"

mdy_hm("01/31/2017 08:01")
# [1] "2017-01-31 08:01:00 UTC"

# you can also force creation of a date-time by supplying a time zone
ymd(20170131, tz = "UTC")
# [1] "2017-01-31 UTC"

# dates from individual components
# instead of a single string sometimes we'll have the indivdual components of our date object
# example:
flights %>% select(year, month, day, hour)
# A tibble: 336,776 x 4
# year month   day  hour
# <int> <int> <int> <dbl>
#         1  2013     1     1  5.00
# 2  2013     1     1  5.00
# 3  2013     1     1  5.00
# 4  2013     1     1  5.00
# 5  2013     1     1  6.00
# 6  2013     1     1  5.00
# 7  2013     1     1  6.00
# 8  2013     1     1  6.00
# 9  2013     1     1  6.00
# 10  2013     1     1  6.00
# # ... with 336,766 more rows



# to create a date/time from this input use make_date for dates and make_datetime for date-times
flights %>% 
        dplyr::select(year, month, day, hour, minute) %>% 
        mutate(
                departure = make_datetime(year, month, day, hour, minute)
        )
# A tibble: 336,776 x 5
# year month   day  hour departure 
# <int> <int> <int> <dbl> <date>    
#         1  2013     1     1  5.00 2013-01-01
# 2  2013     1     1  5.00 2013-01-01
# 3  2013     1     1  5.00 2013-01-01
# 4  2013     1     1  5.00 2013-01-01
# 5  2013     1     1  6.00 2013-01-01
# 6  2013     1     1  5.00 2013-01-01
# 7  2013     1     1  6.00 2013-01-01
# 8  2013     1     1  6.00 2013-01-01
# 9  2013     1     1  6.00 2013-01-01
# 10  2013     1     1  6.00 2013-01-01
# # ... with 336,766 more rows


# defining examples for the rest of the chapter
make_datetime_100 = function(year, month, day, time) {
        make_datetime(year, month, day, time %/% 100, time %% 100)
}



flights_dt = flights %>% 
        filter(!is.na(dep_time), !is.na(arr_time)) %>% 
        mutate(
                dep_time = make_datetime_100(year, month, day, dep_time),
                arr_time = make_datetime_100(year, month, day, arr_time),
                sched_dep_time = make_datetime_100(year, month, day, sched_dep_time),
                sched_arr_time = make_datetime_100(year,month, day, sched_arr_time)
        ) %>% 
        select(origin, dest, ends_with("delay"), ends_with("time"))

flights_dt
# A tibble: 328,063 x 9
# origin dest  dep_delay arr_delay dep_time            sched_dep_time      arr_time           
# <chr>  <chr>     <dbl>     <dbl> <dttm>              <dttm>              <dttm>             
#         1 EWR    IAH        2.00     11.0  2013-01-01 05:17:00 2013-01-01 05:15:00 2013-01-01 08:30:00
# 2 LGA    IAH        4.00     20.0  2013-01-01 05:33:00 2013-01-01 05:29:00 2013-01-01 08:50:00
# 3 JFK    MIA        2.00     33.0  2013-01-01 05:42:00 2013-01-01 05:40:00 2013-01-01 09:23:00
# 4 JFK    BQN       -1.00    -18.0  2013-01-01 05:44:00 2013-01-01 05:45:00 2013-01-01 10:04:00
# 5 LGA    ATL       -6.00    -25.0  2013-01-01 05:54:00 2013-01-01 06:00:00 2013-01-01 08:12:00
# 6 EWR    ORD       -4.00     12.0  2013-01-01 05:54:00 2013-01-01 05:58:00 2013-01-01 07:40:00
# 7 EWR    FLL       -5.00     19.0  2013-01-01 05:55:00 2013-01-01 06:00:00 2013-01-01 09:13:00
# 8 LGA    IAD       -3.00    -14.0  2013-01-01 05:57:00 2013-01-01 06:00:00 2013-01-01 07:09:00
# 9 JFK    MCO       -3.00    - 8.00 2013-01-01 05:57:00 2013-01-01 06:00:00 2013-01-01 08:38:00
# 10 LGA    ORD       -2.00      8.00 2013-01-01 05:58:00 2013-01-01 06:00:00 2013-01-01 07:53:00
# # ... with 328,053 more rows, and 2 more variables: sched_arr_time <dttm>, air_time <dbl>


# visualize times across the year
flights_dt %>% ggplot(aes(dep_time)) +
        geom_freqpoly(binwidth = 86400) # one day is 86400 seconds

# departure time within a single day
flights_dt %>% 
        filter(dep_time < ymd(20130102)) %>% 
        ggplot(aes(dep_time)) +
        geom_freqpoly(binwidth = 600) # 600 seconds = 10 minutes
# note that when you use date-times in a numeric context (like in a histogram ) 1 means one second
# in a date object 1 means 1 day


# Dates from Other Types
# switch between date-time and date
as_datetime(today())
# [1] "2018-03-02 UTC"
as_date(now())
# [1] "2018-03-02"


# non-date objects will fail to parse
ymd(c("2010-10-10", "bananas"))


## Date Time Components
# we know how to get date-time data into R's date-time strucuture
# what can we do with these one we have them in our strucuture?
# we can extract thier components

# getting components
# use year(), month(), mday(), yday(), wday(), hour(), minute(), and second()
datetime = ymd_hms("2016-07-08 12:34:56")

#examples
year(datetime)
# [1] 2016

month(datetime)
# [1] 7

mday(datetime)
# [1] 8

yday(datetime)
# [1] 190

wday(datetime)
# [1] 6

# for month and wday we can set label = T to return abbreviated name of the variable
# set to false for the full name:
month(datetime, label = T)
# [1] Jul
# 12 Levels: Jan < Feb < Mar < Apr < May < Jun < ... < Dec

wday(datetime, label = T, abbr = F)
# [1] Friday
# 7 Levels: Sunday < Monday < Tuesday < ... < Saturday


# we can use wday() to see that more flights depart during the week than on the weekend:
flights_dt %>% 
        mutate(wday = wday(dep_time, label = T)) %>% 
        ggplot(aes(x = wday)) +
        geom_bar()

# there is an interesting pattern if we look at the average depature delay by minute within each hour
# it looks like flights leaving in minutes 20-30 and 50-60 have much lower delays than the rest of the hour!
flights_dt %>% 
        mutate(minute = minute(dep_time)) %>% 
        group_by(minute) %>% 
        summarize(
                avg_delay = mean(arr_delay, na.rm = T),
                n = n()
        ) %>% 
        ggplot(aes(minute, avg_delay)) +
        geom_line()

# interestingly - if we look at scheduled departure time we don't see a strong pattern
sched_dep = flights_dt %>% 
        mutate(minute = minute(sched_dep_time)) %>% 
        group_by(minute) %>% 
        summarize(
                avg_delay = mean(arr_delay, na.rm = T),
                n = n()
        )

ggplot(sched_dep, aes(minute, avg_delay)) +
        geom_line()

# why do we see that pattern with the actual departure times?
# like much data collected by humans there is a bias to have depatrue times at "nice" intervals
ggplot(sched_dep, aes(minute, n)) +
        geom_line()

# rounding with dates
# we can round our dates to a nearby unit of time
# floor_date, round_date, ceiling_date
# these functions take a vector of dates to adjust and then name of the unit to round to
# example: plotting the number of flights per week
flights_dt %>% 
        count(week = floor_date(dep_time, "week")) %>% 
        ggplot(aes(week, n)) +
        geom_line()

# setting components 
# we can also use each accessor function to set the components of a date/time:
(datetime <- ymd_hms("2016-07-08 12:34:56"))
# [1] "2016-07-08 12:34:56 UTC"

year(datetime) <- 2020
datetime
# [1] "2020-07-08 12:34:56 UTC"

month(datetime) <- 01
datetime
# [1] "2020-01-08 12:34:56 UTC"

hour(datetime) <- hour(datetime) + 1
datetime
# [1] "2020-01-08 13:34:56 UTC"

# rather than modifying in place - we can create a new date time object with update()
# this allows us to set multiple values at once
update(datetime, year = 2020, month = 2, mday = 2, hour = 2)
# [1] "2020-02-02 02:34:56 UTC"

# if values are too big they will roll over
ymd("2015-02-01") %>% 
        update(mday = 30)
# [1] "2015-03-02"

ymd("2015-02-01") %>% 
        update(hour = 400)
# [1] "2015-02-17 16:00:00 UTC"

# we can use update to show the distribution of flights across the course of each day for every day of the year
# setting larger components of a date to a constant is a powerful technique
# this allows us to explore patterns in smaller components
flights_dt %>% 
        mutate(dep_hour = update(dep_time, yday = 1)) %>% 
        ggplot(aes(dep_hour)) +
        geom_freqpoly(binwidth = 300)



# Time Spans
# we can do arithmetic with dates and times
# there are three important classes that represent time spans
        # durations = represent an exact number of seconds
        # periods = represent human units like weeks and months
        # intervals = represents a starting and an ending point

# Durations
# when you subtract two dates we get a difftime object
h_age <- today() - ymd(19791014)
h_age
# Time difference of 14021 days

# a difftime class object reocrds a time span of seconds, minutes, hours, days or weeks
# lubridate has a function that always uses seconds in the calculation - the duration
as.duration(h_age)
# [1] "1211414400s (~38.39 years)"

# durations come with a bunch of convenient constructors:
dseconds(15)
# [1] "15s"
dminutes(10)
# [1] "600s (~10 minutes)"
dhours(c(12,24))
# [1] "43200s (~12 hours)" "86400s (~1 days)"
ddays(0:5)
# [1] "0s"                "86400s (~1 days)"  "172800s (~2 days)" "259200s (~3 days)" "345600s (~4 days)"
# [6] "432000s (~5 days)"

dweeks(3)
# [1] "1814400s (~3 weeks)"

dyears(1)
# [1] "31536000s (~52.14 weeks)"

# durations always record the time span in seconds!
# larger units are calculated by counting up the seconds in that duration

# we can multiply durations
2*dyears(1)
# [1] "63072000s (~2 years)"

dyears(1) + dweeks(12) + dhours(15)
# [1] "38847600s (~1.23 years)"

# we can also subtract durations from days
tomorrow <- today()  + ddays(1)
# [1] "2018-03-05"

last_year <- today() - dyears(1)
# [1] "2017-03-04"

# durations represent number of seconds
# we might get unexpected results because of this:
one_pm <- ymd_hms(
        "2016-03-12 13:00:00",
        tz = "America/New_York"
)
one_pm
# [1] "2016-03-12 13:00:00 EST"

one_pm + ddays(1)
# [1] "2016-03-13 14:00:00 EDT"

# one day after march 12 1pm is now march 13 2pm?
# also notice the time zone change!
# this is because of DST - March 12 only has 23 hours - if we add a full day's worth of seconds we get a different time!


# Periods
# to solve this problem above - lubridate has periods
# periods are time spans that do not have a fixed length in seconds
# they work with human time in months and weeks
# they will work in a more intuitive way
one_pm
# [1] "2016-03-12 13:00:00 EST"
one_pm + days(1)
# [1] "2016-03-13 13:00:00 EDT"

# like durations periods are created with a number of constructors
seconds(12)
# [1] "12S"
minutes(10)
# [1] "10M 0S"
hours(c(12, 24))
# [1] "12H 0M 0S" "24H 0M 0S"

days(7)
# [1] "7d 0H 0M 0S"

months(1:6)
# [1] "1m 0d 0H 0M 0S" "2m 0d 0H 0M 0S" "3m 0d 0H 0M 0S" "4m 0d 0H 0M 0S" "5m 0d 0H 0M 0S" "6m 0d 0H 0M 0S"

weeks(3)
# [1] "21d 0H 0M 0S"

years(1)
# [1] "1y 0m 0d 0H 0M 0S"

# we can perform operations on periods
10*(months(6) + days(1))
# [1] "60m 10d 0H 0M 0S"

days(50) + hours(25) + minutes(2)
# [1] "50d 25H 2M 0S"


# leap year
ymd("2016-01-01") + dyears(1)
# [1] "2016-12-31"

ymd("2016-01-01") + years(1)
# [1] "2017-01-01"

# daylight savings time
# duration method
one_pm + ddays(1)
# [1] "2016-03-13 14:00:00 EDT"

# periods method
one_pm + days(1)
# [1] "2016-03-13 13:00:00 EDT"



# let's use periods to fix an oddity in our flights dataset
# some planes appear to have arrived at thier destination before they departed!!
flights_dt %>% 
        filter(arr_time < dep_time)
# # A tibble: 10,633 x 9
# origin dest  dep_delay arr_delay dep_time            sched_dep_time      arr_time           
# <chr>  <chr>     <dbl>     <dbl> <dttm>              <dttm>              <dttm>             
#         1 EWR    BQN        9.00    - 4.00 2013-01-01 19:29:00 2013-01-01 19:20:00 2013-01-01 00:03:00
# 2 JFK    DFW       59.0      NA    2013-01-01 19:39:00 2013-01-01 18:40:00 2013-01-01 00:29:00
# 3 EWR    TPA      - 2.00      9.00 2013-01-01 20:58:00 2013-01-01 21:00:00 2013-01-01 00:08:00
# 4 EWR    SJU      - 6.00    -12.0  2013-01-01 21:02:00 2013-01-01 21:08:00 2013-01-01 01:46:00
# 5 EWR    SFO       11.0     -14.0  2013-01-01 21:08:00 2013-01-01 20:57:00 2013-01-01 00:25:00
# 6 LGA    FLL      -10.0     - 2.00 2013-01-01 21:20:00 2013-01-01 21:30:00 2013-01-01 00:16:00
# 7 EWR    MCO       41.0      43.0  2013-01-01 21:21:00 2013-01-01 20:40:00 2013-01-01 00:06:00
# 8 JFK    LAX      - 7.00    -24.0  2013-01-01 21:28:00 2013-01-01 21:35:00 2013-01-01 00:26:00
# 9 EWR    FLL       49.0      28.0  2013-01-01 21:34:00 2013-01-01 20:45:00 2013-01-01 00:20:00
# 10 EWR    FLL      - 9.00    -14.0  2013-01-01 21:36:00 2013-01-01 21:45:00 2013-01-01 00:25:00
# # ... with 10,623 more rows, and 2 more variables: sched_arr_time <dttm>, air_time <dbl>

# these are overnight flights...
# we used the same date information for both the departure and arrival times but these flights arrived on the following day
# we can fix this by adding days(1) to the arrival time of each overnight flight
flights_dt <- flights_dt %>% 
        mutate(
                overnight = arr_time < dep_time,
                arr_time = arr_time + days(overnight*1),
                sched_arr_time = sched_arr_time + days(overnight*1)
        )

# now all of our flights are fixed!!
flights_dt %>% 
        filter(overnight, arr_time < dep_time)
# A tibble: 0 x 10
# ... with 10 variables: origin <chr>, dest <chr>, dep_delay <dbl>, arr_delay <dbl>, dep_time <dttm>,
#   sched_dep_time <dttm>, arr_time <dttm>, sched_arr_time <dttm>, air_time <dbl>, overnight <lgl>




# intervals
# the nature of dates gives us some weird information
# for example years(1) / days(1) depends on what year it is!
# 2015 = 365 days and 2016 = 366!
# we don't have quit enough information for lubridate to give us a clear answer!
# we get a warning when we run into situations like this

years(1) / days(1)
# estimate only: convert to intervals for accuracy
# [1] 365.25

# if we want a more accurate estimate we will have to use Intervals!
# an interval is a duration with a starting point
# this makes it precise 
next_year <- today() + years(1)
(today() %--% next_year) / ddays(1)
# [1] 365

# to find out how many periods fall into an interval we need to use integer division
(today() %--% next_year) %/% days(1)
# [1] 365

# how do we pick between durations, periods and intervals?
# always go with the simplest solution to solve your problem
# physical time = durations
# human time = PERIODS
# time spans from starting point? = INTERVALS


# quick notes on time zones
# time zones are extrememly complicated
# they pose a few challenges in our data analysis
# R will use the standard IANA time zones to avoid naming confusion (there are many ESTs)
# IANA use the "/" format <continent> / <city>
# we can find out what R thinks our timezone is with Sys.timezone
Sys.timezone()
# [1] "America/Los_Angeles"

# we can see the complete list of time zones names with OlsonNames()
length(OlsonNames())
# [1] 592
head(OlsonNames())
# [1] "Africa/Abidjan"     "Africa/Accra"       "Africa/Addis_Ababa" "Africa/Algiers"     "Africa/Asmara"     
# [6] "Africa/Asmera" 


# in R the time zone is an attribute of the date-time that only controls printing

# unless otherwise specified lubridate always used UTC (Coordinated Universal Time)
# this is the standard time zone used by the scientific community and close to GMT
# it does not have DST
# operations that combine date-times like c() will often drop the time zone
# in this case the date-times will display in your local time zone
# we can change the time zone in two ways
        # keep the instant in time the same and change how it's displayed
        # change the underlying instant in time!





## Part 3: Program
# in this part of the book we will improve our programming skills
# programming is an essential skill for all data science work
# you must use your computer to do data science
# programming produces code and code is a tool of communication
# the code tells the computer what to do!
# code can also communicate to other humans!
# you must think about code as a means of communication to other humans!!
# THINK OF THE FUTURE YOU WHEN YOU CODE!
# writing clear code is important to help users understand what you did 
# GETTING BETTER AT PROGRAMMING ALSO INVOLVES GETTING BETTER AT COMMUNICATION!!!
# overtime we want our code to be easier to read
# writing code is similiar to writing prose
# re-writing is the key to clarity
# after you solve a complex challenge you need to review your code and try to make it as clear as possible
# in the following four chapters we will learn skills to tackle new programs and solve existing ones with ease:
        # Chapter 14: deep dive into the %>%  pipe operator
        # Chapter 15: functions - you should never copy and paste more than twice!!!
        # Chapter 16: data structures - how to understand lists vs. data frames
        # Chapter 17: interation - similiar things again and again based on different inputs!! FUNCTIONAL PROGRAMMING!
# learning more about programming will help you solve programs quicker!
# learning programming is a long term plan!!
# check out advanced R as another resource!!










# chapter 14 pipes --------------------------------------------------------





## Chapter 14: Pipes with magrittr
# pipes are powerful tools for clearly expressing a sequence of multiple operations
# this chapter explores pipes in more detail
# the pipe comes from the magrittr package and comes automatically in tidyverse
library(tidyverse)

# the point of the pipe is to help you write code in a way that is easier to read and understand
# to explore why the pipe is useful we will experiment different ways to write the same code

# Bunny Foo Foo example:
# using objects and verbs we can re-write the bunny foo foo story in code
        # save each intermediate step as a new object
        # overwrite the original object many times
        # compose functions
        # use the pipe
foo_foo <- little_bunny()


## intermediate coding
# the simplest approach is to save each step as a new object:
foo_foo_1 <- hop(foo_foo, through = forest)
foo_foo_2 <- scoop(foo_foo_1, up = mice)
foo_foo_3 <- bop(foo_foo_2, on = head)

# the main downside of this form is that it forces you to name each intermediate element
# if there are natural names then this might be a good use case
# but many times there aren't natural names and we add a list of numeric suffixes to make the names unqiue
# this leads to two problems:
        # the code is cluttered with unimportant names
        # you have to carefully increment the suffix on each line
# we can easily mess up the "overwriting" of the previous line and have our code break
# also creating many intermediate steps creates many copies of this variable - this takes up space!!
# WORRY ABOUT MEMORY WHEN IT BECOMES A PROBLEM
# R isn't stupid it will share columns across data frames where available

# memory example
library(pryr)
data("diamonds")
diamonds2 <- diamonds %>% 
        dplyr::mutate(price_per_carat = price / carat)

pryr::object_size(diamonds)
# 3.46 MB

pryr::object_size(diamonds2)
# 3.89 MB

pryr::object_size(diamonds, diamonds2)
# 3.89 MB

# note diamonds takes up 3.46 MB
# note diamonds2 takes up 3.89 MB
# together diamonds and diamonds2 take up 3.89 MB!!!
# diamonds2 has 10 columns in common with diamonds and there is no need to duplicate all that data
# these variables will be shared between the two dataframes
# the variables will be shared unless we modify one of the variables

diamonds$carat[1] <- NA

pryr::object_size(diamonds)
# 3.46 MB

pryr::object_size(diamonds2)
# 3.89 MB

pryr::object_size(diamonds, diamonds2)
# 4.32 MB




## overwriting the original coding
# instead of creating intermediate objects at each step we can overwrite the original object
foo_foo <- hop(foo_foo, through = forest)
foo_foo <- scoop(foo_foo, up = mice)
foo_foo <- bop(foo_foo, on = head)

# this takes less typing and less thinking and we possibly will make less mistakes
# however there are still problems:
        # debugging is painful - if there is a mistake we must "pull the string" all the way to the first object
        # repeating the object "hides" some of the transformation going on with each line


## function composition
# another approach is to abandon assigment a just string together the function calls together
bop(
        scoop(
                hop(foo_foo, through = forest),
                up = mice
        ),
        on = head
)


# here the disadvantage is that we have to read from inside out
# arugments are spread far apart - Dagwood Sandwich
# this is hard for a human to look at and determine what is going on


## Pipe Coding
foo_foo %>% 
        hop(through = forest) %>% 
        scoop(up = mice) %>% 
        bop(on = head)

# this form focuses on verbs not nouns
# we can read this as a series of function compositions like its a set of actions
# foo = hops <then> scoops <then> bops
# the one downside is we have to known what %>%  does - THE PIPE

# the pipe works by performing a "lexical transformation" 
# behind the scenes magrittr reassembles the code in the pipe to a form that works by overwritting intermediate objects
# example of what magrittr does: eseentially creates a custom function in the back -end that strings our pipes together!!
my_pipe <- function(.) {
        . <- hop(., through = forest)
        . <- scoop(., up = mice)
        bop(., on = head)
}
my_pipe(foo_foo)

# however this means that the pipe won't work for two classes of functions:
# functions that use the current envrioment
# for example: assign() will create a new variable with the given name in the current enviroment
assign("x", 10)
x
# [1] 10

x %>% assign(100)
x
# Error in assign(., 100) : invalid first argument

# the use of assign with the pipe does not work because it assigns it to a temporary enviroment used by %>% 
# if we want to use assign within the pipes we need to specify the enviroment explicitly
env <- environment()
"x" %>% assign(100, envir = env)
x
# [1] 100


# other functions with this problem include get() and load()
# functions that use lazy evaluation will not work in pipes
# in R - function agruments are only computed when the function uses them not prior to calling the function!
# the pipe computes each element in turn so we can't rely on this behavior

# one place that this is a problem is tryCatch() which lets you capture and handle errors
tryCatch(stop("!"), error = function(e) "An error")
# [1] "An error"

stop("!") %>% 
        tryCatch(error = function(e) "An error")
# Error in eval(lhs, parent, parent) : !

# there are a relatively wide class of functions with this behavior including:
# try(), suppressMessages(), suppressWarnings()


## When not to use the PIPE?
# the pipe is a great tool but does not solve every problem
# pipes are most useful for rewriting a fairly short linear sequence of operations
# we should check for another tool when:
        # pipes are longer than 10 steps: use intermediate objects with meaningful names
        # you have multiple inputs or outputs
        # you are starting to think about a directed graph with a comples dependency structure
# pipes are fundamentally linear and expressing comples relationships with them will give back confusing code

## Other Tools from magrittr
# all packages from tidyverse automatically make %>%  available
#  there are some useful tools in magrittr explictly that may be useful

# when working with more comples pipes it's sometimes useful to call a function for its side effects
# maybe you want to print out the current object or plot it or save it to the disk
# many times such functions don't return anything effectively terminating the pipe

# to work around this problem you can use the "tee" pipe
# %T>%  works like %>% expect that it returns the lefthand side instead of the righthand side
# it's called "tee" because it's like a lteral T-shaped pipe
library(magrittr)


rnorm(100) %>% 
        matrix(ncol = 2) %>% 
        plot() %>% 
        str()
# NULL

# using t pipe to plot the results and then run the str function!!
rnorm(100) %>% 
        matrix(ncol = 2) %T>%
        plot() %>% 
        str()
# num [1:50, 1:2] 1.474 0.677 0.38 -0.193 1.578 ...


# if you are working with functions that don't have a data frame based API
# i.e. you pass them indivdual vectors not a data frameand expressions to be evaluted in the context of that data frame
# we can use the dollar pipe %$%
# this "exploded" out the variables in a data frame so that you can refer them explicitly
# this is useful when working with many functions in base R:
mtcars %$%
        cor(disp, mpg)
# [1] -0.8475514


# for assignment magrittr provides the %<>% operator which allows you to replace code like:
mtcars <- mtcars %>% 
        transform(cyl = cyl *2)

# magrittr example
mtcars %<>% transform(cyl = cyl * 2) 




# chapter 15 functions ----------------------------------------------------




## Chapter 15: Functions
# one of the best ways to improve your reach as a data scientist is to write functions
# functions allow you to automate common tasks in a powerful and general way other than copying / pasting
# writing a function has three big advantages:
        # you can give a function a name that makes your code easier to understand
        # as requirements change you only need to update the code in one place instead of everywhere
        # you eliminate the chance of making incidental mistakes when you copy and paste
# writing good functions is a lifetime journey
# goal of this chapter is to get started on the journey of writing functions

# When should I write a function?
# you should consider writing a function whenever you've copied and pasted a block of code more than twice
# for example take a look at this code...what does it do?
df <- tibble::tibble(
        a = rnorm(10), 
        b = rnorm(10),
        c = rnorm(10),
        d = rnorm(10)
)

df$a <- (df$a - min(df$a, na.rm = T)) / (max(df$a, na.rm = T) - min(df$a, na.rm = T))
df$b <- (df$b - min(df$b, na.rm = T)) / (max(df$b, na.rm = T) - min(df$b, na.rm = T))
df$c <- (df$c - min(df$c, na.rm = T)) / (max(df$c, na.rm = T) - min(df$c, na.rm = T))
df$d <- (df$d - min(df$d, na.rm = T)) / (max(df$d, na.rm = T) - min(df$d, na.rm = T))

# this code re-scales each column to have a range of 0 to 1
# this is very error prone
# writing a function will eliminate the need to rewrite each step of this code

# to write a function we need to analyze the code: how many inputs does it have?
(df$a - min(df$a, na.rm = T)) /
        (max(df$a, na.rm = T) - min(df$a, na.rm = T))
# [1] 0.0000000 0.8331891 0.4486110 0.2612747 1.0000000 0.1491435 0.5276978 0.8728839 0.6331408 0.3123453

# this code only has one input df$a
# to make the inputs more clear it is good idea to rewrite the code using temporary variable names with general names
# our code only requires a single numeric vector so we can call it x
x <- df$a
(x - min(x, na.rm = T)) / (max(x, na.rm = T) - min(x, na.rm = T))
# [1] 0.0000000 0.8331891 0.4486110 0.2612747 1.0000000 0.1491435 0.5276978 0.8728839 0.6331408 0.3123453

# there is still some duplication in this code
# we are computing the range of the data three times when we can do this in one step
rng <- range(x, na.rm = T)
(x - rng[1]) / (rng[2] - rng[1])
# [1] 0.0000000 0.8331891 0.4486110 0.2612747 1.0000000 0.1491435 0.5276978 0.8728839 0.6331408 0.3123453

# pulling out intermediate claclulations into named variables is a good practice
# it makes it more clear what the code is doing
# now we can build a function with this code
rescale01 <- function(x) {
        rng <- range(x, na.rm = T)
        (x - rng[1]) / (rng[2] - rng[1])
}

rescale01(c(0,5,10))
# [1] 0.0 0.5 1.0

# there are three key steps to creating a new function:
        # you need to pick a name for the function
        # you list the inputs or arguements to the function inside function(...)
        # you place the code you have developed in the body of the function a {...} block of code
# note the overall process:
# we only made a function after we figure out how to make it work with one simple input
# it's easier to start with working code and turn it into a function
# it's harder to create a function and then try to make it work

# let's check our function with different inputs
rescale01(c(-10,0,10))
# [1] 0.0 0.5 1.0

rescale01(c(1,2,3,NA,5))
# [1] 0.00 0.25 0.50   NA 1.00

# as we write more and more functions we will eventually want to convert these informal intervative tests to formal automated tests
# formal automated tests are called UNIT TESTING

# we can now simplify the original example now that we have our rescale function
df$a <- rescale01(df$a)
df$b <- rescale01(df$b)
df$c <- rescale01(df$c)
df$d <- rescale01(df$d)

# compare to the original this code is easier to understand and we've eliminated a lot of copy pasting
# we still have some duplication

# another advantage to functions is that if our requirements change we only need to make the change in one place!!
# for example we might discover that some of our values include infinite values: causing our function to fail
x <- c(1:10, Inf)
rescale01(x)
# [1]   0   0   0   0   0   0   0   0   0   0 NaN

# because we extracted the code into a function we only need to fix it in one place
rescale01 <-  function(x) {
        rng <- range(x, na.rm = T, finite = T)
        (x - rng[1]) / (rng[2] - rng[1])
}

rescale01(x)
# [1] 0.0000000 0.1111111 0.2222222 0.3333333 0.4444444 0.5555556 0.6666667 0.7777778 0.8888889 1.0000000
# [11]       Inf

# this is an important part of "do not repeat yourself" DRY principal
# the more repition you have in your code the more places we need to remember to update!!
# this will likely create bugs over time!! DO NOT REPEAT YOURSELF!!!

# functions are for humans and computers
# remember that functions are not just for the computer but also for humans
# R doesn't care what your function is called or what comments it contains
# these are important for human readers!!
# this section discusses things you should keep in mind when writing functions for human readers

# the name of your function is important!!
# should be short, and clearly evoke what the function does
# it is better to be clear than short

# functino names should be VERBS!!!
#too short
f()
# not a verb, or descriptive
my_awesome_function()
# long but not clear
impute_missing()
collapse_years()

# if your function name is composed of multiple words - use SNAKE CASE
# i.e. snake_case each word is sperated by an underscore
# camelCase is a popular alternative
# the important thing is to be consistent with what ever you decidie
# R itself is not very consistent - don't fall into the same trap by making your code as consistent as possible

# never do this!
col_mins <- function(x,y) {}
rowMaxes <- function(x,y) {}


# if you have a family of functinos that do simliar things make sure they have consistent names and agruements
# use a common prefix to indicate they are connected
# prefixes are better than suffixes!!! R uses autocomplete

# good
input_select()
input_checkbox()
input_text()

# not good at all
select_input()
checkbox_input()
text_input()

# a good example of this design is the stringr package
# if we don't remember what function we need we can start typing str_ and a list of all available functions will appear

# aviod overwriting other functions and variables
# try to avoid naming functions the same as an already existing function

# don't do this!
T <- FALSE
c <- 10
mean <- function(x) sum(x)


# use comments to explain the "why" of your function
# you should aviod comments that explain the "what" or "how"
# we should be able to understand from the code the "what" and the "how"
# code can never caputre the reasoning behind the deciisions
# why did we choose this approach?
# what else did we try that did not work?

# we can also use commnets to break up our file into easily readable code chunks
# there is a navigator at the bottom of the editor!!! VVVVVVVV

# load data ---------------------------------------------------------------


# plot data ---------------------------------------------------------------


# Conditional Execution
# an if statement allows you to conditionally execute code
# it looks like this:
if (condition) {
        # code executed when condition is TRUE
} else {
        # code executed when the condition in FALSE
}

# to get help on if we need backticks ?`if`
# here is a simple function that uses an if statement
# the goal of this function is to return a logical vector describing whether or not each element of a vector is named
has_name <- function(x) {
        # define inputs
        nms <- names(x)
        
        # define if statement
        if(is.null(nms)) {
                # code if true
                rep(FALSE, length(x))
                
        } else {
                # code if false
                !is.na(nms) & nms != ""
        }
}

# this function takes advantage of the standard return rule:
# A FUNCTION RETURNS THE LAST VALUE THAT IT COMPUTED
# in this case it is one of the two branches of our if statement


# Conditions
# the condition must evaluate to either TRUE or FALSE
# if its a vector we will get a warning message
# if it is an NA we will get an error
# watch out for these messages in our own code
if(c(TRUE, FALSE)) {}
# Warning message:
#         In if (c(TRUE, FALSE)) { :
#                         the condition has length > 1 and only the first element will be used
if (NA) {}
# Error in if (NA) { : missing value where TRUE/FALSE needed


# we can use || or && (or / and) to combine multiple logical expressions
# these operators are "short circuiting" 
# as soon as || sees the first true it return TRUE without computing anything else
# as soon as && sees the first FALSE it returns FALSE
# you should never use | or & (single characters) in an IF statement:
# these are vectorized operations that apply to multiple values (this is why we can use them in filter())
# if we do have a logical vector we can use any() or all() to collapse into a single value

# be careful when testing for equality
# "==" is vectorized which means it's easy to get more than one output
# either check the length is already 1, collapse with all() or any() or use the non-vectorized identical()
# indentical() is very strict: it always returns a single true or a single false
# it also does not coerce types: be careful comparing integers to doubles
identical(0L, 0)
# [1] FALSE

# also be careful with floating point numbers
x <- sqrt(2) ^2
x
# [1] 2

x == 2
# [1] FALSE

x - 2
# [1] 4.440892e-16

# instead try to use dplyr::near() for comparisions as described in comparisons


# Multiple Conditions
# we can chain multiple if statements together
if (this) {
        # do that
} else if (that) {
        # do something else
} else {
        # final statement
}

# if you end up with a very long series of chained if statements we should consider rewriting the statement
# one useful technique is the switch() function
# it allows you to evaluate code based on position name
function(x, y, op) {
        switch(op,
               plus = x+y,
               minus = x - y,
               times = x*y,
               divide = x / y,
               stop("Unknown op!"))
}

# another useful function is cut() 
# it is used to discretize continous variables

# Code Style
# both if and functino should almost always be followed by squiggly brackets {}
# and the contents should be indented by two spaces
# this makes it easier to see the hierarchy in your code by skimming the lefthand margin
# an opening curly brace should never go on its own line and should always be followed by a new line
# A closing curly brace should always go on its own line unless followed by an else()
# always indent code inside curly brances

# good
if (y < 0 && debug) {
        message("Y is negative")
}

if (y == 0) {
        log(x)
} else {
        y ^ x
}


# bad
if (y < 0 && debug)
message("Y is negative")

if (y == 0) {
        log(x)
} else {
y^x
}

# it's ok to drop the curly braces if you have a very short if statement that you can do on one line
y <- 10
x <- if(y <20) "Too low" else "too high bitch"

# full form is easier to read for many other cases

if (y <20) {
        x <- "too low"
} else {
        x <- "too high bitch"
}



## Function Agruements
# the agruments to a function typically fall into two broad sets:
        # one set supplies the data to compute on 
        # the other supplies agruements that control the details of the computation
# example:
        # in the log(), the data is x, the the detail is the base of the logarithm
        # in mean(), the data is x, and the details are how much data to trim and how to handle missing values
        # in t.test(), the data are x and y, and the details of the test are alternative, mu, paired, var.equal conf.level
        # in str_c() - we can simply supply any number of strings to ... and the concatenation are controlled by sep and collapse
# the data arguements should come first
# detail arguments should go at the end
# you specify a default value in the same way you call a function with a named arguement

# compute confidence interval around
# mean using normal approximation
mean_ci <- function(x, conf = .95) {
        se <- sd(x) / sqrt(length(x))
        alpha <- 1 - conf
        mean(x) + se * qnorm(c(alpha / 2, 1 - alpha / 2))
}

# test out our function
x <- runif(100)
mean_ci(x)
# [1] 0.5035990 0.6208593

mean_ci(x, conf = .99)
# [1] 0.4851761 0.6392822

# our default value should almost always be the most common value
# the few exceptions have to deal with safety
# it makes sense to have the default of na.rm = FALSE because omitting missing values could cause problems
# na.rm = TRUE is what we would normally put in our code - it is better for warnings to pop when we do encounter missing values

# when you call a function - we typically omit the names of the data arguements because they are used commonly
# if we override the default value of a detail arguement we should use the full name

# good
mean(1:10, na.rm = T)
# [1] 5.5

# bad
mean(x = 1:10, FALSE)
mean(TRUE, x = c(1:10, NA))

# this is a lot of extra work for little additional gain
# a useful compromise is the built-in stopifnot() function
# this checks that each agruement is TRUE and produces a generic error message if not
wt_mean <- function(x, w, na.rm = F) {
        stopifnot(is.logical(na.rm), length(na.rm) == 1)
        stopifnot(length(x) == length(w))
        
        if(na.rm) {
                miss <- is.na(x) | is.na(w)
                x <- x[!miss]
                w <- w[!miss]
        }
        
        sum(w*x) / sum(x)
}

wt_mean(1:6,6:1, na.rm = "foo")
# Error: is.logical(na.rm) is not TRUE 

# in this case we need to assert what should be true rather than stopping for what might be wrong


## Dot Dot Dot (...)
# many functions in R take an arbitrary number of inputs:
sum(1,2,3,4,5,6,7,8,9,10)
# [1] 55

stringr::str_c("a","b","c","d","e")
# [1] "abcde"

# how does these functions work?
# they all rely on a special arguement: ...
# the special arguement captures any number of agruements that aren't otherwise matched
# THIS IS USEFUL BECAUSE WE CAN SEND THOSE ... TO ANOTHER FUNCTION

# for example we can create wrapper functions around other functions to make our lives easier
# also called "helper" functions
commas <- function(...) stringr::str_c(..., collapse = ",")

letters[1:10]
# [1] "a" "b" "c" "d" "e" "f" "g" "h" "i" "j"

commas(letters[1:10])
# [1] "a,b,c,d,e,f,g,h,i,j"

rule <- function(..., pad = "-") {
        
        title <- paste0(...)
        width <- getOption("width") - nchar(title) - 5
        cat(title, stringr::str_dup(pad, width), "\n", sep = "")
}

rule("Important output")
# Important output-------------------------------------------------------------------------------------

# here ... lets us forward on any arguements that I don't want to deal with to str_c
# this is a great function writing technique
# however be careful: any misspelled arguements will not raise an error!
# typos can easily go unnoticed
x <- c(1,2)
sum(x, na.mr = T)
# [1] 4


## Lazy Evaluation
# arguements in R are lazily evaluated: THEY ARE NOT COMPUTED UNTIL THEY ARE NEEDED
# that means if they are never used they are never called
# this is an important property of R as a programming language


## Return Values
# figureing out what your function should return is usually straightforward: we know why we created the function - to give us a specific output
# there are two things we should consider when returning a value:
        # does returning early make your function easier to read?
        # can you make your function pipeable?

# Explicit Return Statements
# the value returned by the function is usually the last statement it evaluates
# we can choose to return early by using return()
# save the use of return() to signal that you can return early with a simpler solution
# a common reason to do this is because inputs are empty:
complicated_function <- function(x, y, z) {
        if(length(x) == 0 || length(y) == 0) {
                return(0)
        }
        
        # complicated code here
        # we can exit this function if we find length of x and y are equal to 0
}

# another reason is to combine if statements with a complex block and a simple block
# we want to strucutre our blocks simplest to most complex
# we can take advantage of return() to halt the process with a value of the simpler calculation if all criteria are met
f <- function() {
        if(!x) {
                return(something_short)
        }
        
        # do your complex if statement steps here
        # we can stop calculating with we meet the something short criteria
        # else continue on to the more complex block of code
        # this will make our code easier to understand
}



## Writing Pipeable Functions
# if you want to write your own pipeable functions thinking about the return value is important
# there are two main types of pipeable functions:
        # transformation
        # side-effect
# transformation: there is a clear "primary" object passed in the first agruement and we return a modified version
# side effect: are called to perform an action like drawing a plot or saving a file but not transforming an object
# these function should "invisibly" return the first agruement so they are not printed by default
# we can still use these in our pipeline!
# example: function below prints out the number of missing values in a data frame
show_missing <- function(df) {
        n <-  sum(is.na(df))
        cat("Missing values: ", n, "\n", sep = "")
        
        invisible(df)
}

# if we call this function interactively the df object will not get printed out!!
x <- show_missing(mtcars)
# Missing values: 0

# but the data frame is still there!! 
# it is just not printed by default!!
class(x)
# [1] "data.frame"

dim(x)
# [1] 32 11


# WE CAN STILL CALL IT IN A PIPE!!!
mtcars %>% 
        show_missing() %>% 
        mutate(mpg = ifelse(mpg < 20, NA, mpg)) %>% 
        show_missing()
# Missing values: 0
# Missing values: 18


## Enviroment
# the last component of a function is its enviroment
# the enviroment of a function controls how R finds the value associatied with a name
# example:
f <- function(x) {
        x + y
}

# in many programming languages this would be an error because we have not explicitly called what y is!!!
# y is not defined inside of our function!!
# this is valid in R because of lexical scoping to find the value associated with a name
# since y is not defined in the function - it will search throughout the entire enviroment to find the named value!!
y <- 100
f(10)
# [1] 110

y <- 1000
f(10)
# [1] 1010

# this behavior seems like a recipe for bugs and we should avoid creations like this deliberately
# make sure to restart R to a clean state
# the advantage of this behavior is that from a language standpoint it allows R to be very consistent
# every name is looked up using the same set of rules
# for f() this includes the behavior of two things that we might not expect:
        # { and +
# this allows us to do things like
`+` <- function(x, y) {
        if (runif(1) < 0.1) {
                sum(x,y)
        } else {
                sum(x,y) * 1.1
        }
}

table(replicate(1000, 1+2))
# 3 3.3 
# 103 897
rm(`+`)

# this is a common phenomenon in R
# R places few limits on your power
# you can do many things you can't do in other programming languages
# you can also do many things that 99% of the time are extremely ill-advised!!
# example above OVERRIDES HOW ADDITION WORKS!!
# this power and flexibility make tools like ggplot2 and dplyr possible




# chapter 16 vectors ------------------------------------------------------




## Chapter 16: Vectors
# as we start to write our own functions we need to start learning about vectors
# vectors are objects that underlie tibbles
# this is the traditional way to learn R
# most R resources start with vectors and then work thier way up to vectors
# we start with tibbles and then work our way into the underlying structure of vectors
# most of the functions we will write will work with vectors
# the focus of this chapter is on base R data structures
# we will use some examples in purr package 
library(purrr); library(tidyverse)

# Vector Basics:
# there are two types of vectors
        # atomic vectors: logical, integer, double, character, complex, raw
        # lists: recursive vectors - lists can contain other lists
# the main difference between atomic vectors and lists is that atomic vectors are homogeneous, lists are heterogenous
# NULL is meant to represent the abscence of a vector
# NA is used to represent the abscence of a value of a vector
# NULL typically behaves like a vector of length 0

# each vector has two key properties:
        # its type
typeof("letters")
# [1] "character"
typeof(1:10)
# [1] "integer"
        # its length
x <- list("a", "b", 1:10)
length(x)
# [1] 3

# vectors can also contain abritray additional metadata in the form of attributes
# these attributes are used to create augmented vectors
# there are four types of augmented vector:
        # factors are built on top of integer vectors
        # dates and date-times are built on top of numeric vectors
        # data frames and tibbles are built on top of lists

# Important Types of Atomic Vectors
# the four most important types of atomic vector are logical, integer, double and character
# raw and complex are typically not used in data analysis

# Logical Vectors
# logical vectors are the simplest type of atomic vectors because they can take only three values: FALSE, TRUE, NA
# logical vectors are usually constructed with comparison operators
# we can create them by hand with c()
1:10 %% 3 == 0
# [1] FALSE FALSE  TRUE FALSE FALSE  TRUE FALSE FALSE  TRUE FALSE
c(T, T, F, NA)
# [1]  TRUE  TRUE FALSE    NA

# Numeric Vectors
# integer and double vectors are known as numeric vectors
# in R numbers are doubles by default
typeof(1)
# [1] "double"
typeof(1L)
# [1] "integer"

# the distinction between integers and doubles is not usually important
# doubles are approximations - floating point numbers that cannot always be precisely represented with a fixed amount of memory
# all doubles are approximations
x <- sqrt(2) ^2
x
# [1] 2
x - 2
# [1] 4.440892e-16

# this behavior is common when working with floating-point numbers: most calculations include some approximation error
# instead of comparing floating-point numbers using == we should use dplyr::near()
# integers have one special value NA
# doubles have four special values: NA, NaN, Inf and -Inf
c(-1,0,1) / 0
# [1] -Inf  NaN  Inf

# avoid using == to check for these other special values
# instead we use helper functions is.finite(), is.infinite() and is.nan()


# Character Vectors
# character vectors are the most complex type of atomic vector
# each element in a character vector is a string and a string can contain any amount of data
# R uses a global string pool
# each unique string is only stored in memory once and every use of that string points to that representation
# this reduces the amount of memory needed by duplicated strings
x <- "this is a really long string"
pryr::object_size(x)
# 120 B

y <- rep(x, 1000)
pryr::object_size(y)
# 8.11 kB

# y does not take up as much memory as x because each element of y is just a pointer to that same string!!


# Missing Values
# note that each type of atmoic vector has its own missing peice
# all will return the value of NA

NA # logical
# [1] NA
NA_integer_ # integer
# [1] NA
NA_real_ # double
# [1] NA
NA_character_ # character
# [1] NA


## Using Atomic Vectors
# now that we understand the different types of atomic vectors we can review the tools to work with them
# how to we convert on type to another?
# how to tell if an object is a specific type of vector?
# what happens when we work with vectors of different lengths?
# how to name the elements of a vector?
# how to pull out elements of interest?

# Coercion
# there are two ways to convert or coerce one type of vector to another
# explict coercion happens when you call a function to do the conversion i.e. as.integer(), as.character() etc.
# whenever we use explict coercion we should check to see if we can fix the problem upstream!!!
# we could use readr col_types to input the data as the correct format in the original read in the data step!!
# implicit coercion happens when you use a vector in a specific context that expects a type of vector
# example: you use a logical vector with a numeric summary function
# explict coercion should be used rarely!!!

# the most important type of implicit coercion is using a logical vector in a numeric context!!
# TRUE is converted to 1 and FALSE is converted to 0
# that means the sum of a logical vector is the number of trues and the mean is the proportion of trues!!!
x <- sample(20, 100, replace = T)
y <- x > 10
sum(y)
# [1] 46
mean(y)
# [1] 0.46

# we can convert from numeric to logical
if (length(x)) {
        # do something
}

# in this case, 0 is converted to FALSE and everything else is converted to TRUE
# we can be explicit in this implicit coercion by using length(x) > 0

# what happens when you try and create a vector containing multiple types with c()?
# THE MOST COMPLEX TYPE ALWAYS WINS!
typeof(c(NA, 1L))
# [1] "integer"
typeof(c(1L, 1.5))
# [1] "double"
typeof(c(1.5, "a"))
# [1] "character"

# an atomic vector cannot have a mix of different types because the type is a property of a complete vector
# not the individual elements
# if you need to mix multiple types in the same vector we should use a list!!


# Test Functions
# sometimes we want to do different things based on the type of vector
# one option is to use typeof()
# another is to use a test function that returns a TRUE or FALSE
# we can use is_* functions in purrr to test cetain objects for thier type and return T or F based on the function
is_logical() # true for logical
is_double() # true for int
is_numeric() # true for dbl and int
is_character() # true for chr
is_atomic() # true for logical, int, dbl, char
is_list() # true for list
is_vector() # true for all!!!

# each comes with a "scalar" version too


# Scalars and Recyling Rules
# as well as implicitly coercing the types of vectors to be compatible R will implicity coerce the length of vectors
# this is called vector recycling - the shorter a vector is repeated to the same length as the longer vector
# this is useful when you are mixing vectors and "scalars"
# R doesn't really have scalars
# a single number is a vector of length 1
# most built in functions are vectorized - they will operate on a vector of numbers
sample(10) + 100
# [1] 107 104 108 101 110 103 105 109 102 106
runif(10) > .5
# [1]  TRUE FALSE  TRUE FALSE  TRUE  TRUE FALSE FALSE FALSE FALSE

# in R basic mathmatical operations work with vectors
# that means you should never need to perform explicit iteration when performing simple math computations

# what happens when we add to vectors of different lengths?
1:10 + 1:2
# [1]  2  4  4  6  6  8  8 10 10 12
# here R will expand the shortest vector to the same length as the longest = RECYCLING!!
# this is silent except when the length of the longer is not an integer multiple of the length of the shorter
1:10 + 1:3
# [1]  2  4  6  5  7  9  8 10 12 11
# Warning message:
#         In 1:10 + 1:3 :
#         longer object length is not a multiple of shorter object length


# while vector recycling can be used to create very succinct clever code
# this can also silently hide problems
# tidyverse will throw errors when you recycle anything other than a scalar
# we need to use rep() if we want to recycle ourself
tibble(x = 1:4, y = 1:2)
# Error: Column `y` must be length 1 or 4, not 2

tibble(x = 1:4, y = rep(1:2, 2))
# # A tibble: 4 x 2
#        x     y
#      <int> <int>
# 1     1     1
# 2     2     2
# 3     3     1
# 4     4     2

tibble(x = 1:4, y = rep(1:2, each = 2))
# A tibble: 4 x 2
#       x     y
#     <int> <int>
# 1     1     1
# 2     2     1
# 3     3     2
# 4     4     2



# Naming Vectors
# all types of vectors can be named
# we name them with the creation of c()
c(x = 1, y = 2, z = 4)
# x y z 
# 1 2 4 

# we can also set names after the fact with purrr:set_names()
set_names(1:3, c("a","b","c"))
# a b c 
# 1 2 3 

# named vectors are useful for subsetting described next

# Subsetting
# we've used dplyr::filter() to filter rows in a tibble
# filter only with with a tibble or data.frame
# we need a new tool to vectors
# [] is our subsetting function "brakcets"
# example: x[a]
# there are four types of things that you can subset a vector with:
        # a numeric vector containing only integers
        # subsetting with positive intergers gives us the value at the position of our subset character
x <- c("one","two", "Three", "four")
x[c(3,2,4)]
# [1] "Three" "two"   "four" 

# by repeating a position we can make an outpt longer than the input
x[c(1,1,1,1,1,4)]
# [1] "one"  "one"  "one"  "one"  "one"  "four"

# negative values drop the elements at the specified positions:
x[c(-1,-3,-4)]
# [1] "two"

# it is an error to mix positive and negtive values
x[c(-1,1)]
# Error in x[c(-1, 1)] : only 0's may be mixed with negative subscripts

# subsetting with a logical vector keeps all values corresponding to a TRUE value!!
# this is most useful in also using comparision functions
x <- c(10,3,NA,5, 8, 1, NA)

# all non-missing values of X
# this reads - give me values for which x is not NA = T
x[!is.na(x)]
# [1] 10  3  5  8  1

# all even or missing values of x
x[x %% 2 == 0]
# [1] 10 NA  8 NA

# if you have a named vector you can subset it with a character vector:
x <- c(abc = 1, def = 2, xyz = 5)
x[c("xyz", "def")]
# xyz def 
# 5   2 

# like with positive integers you can also use a character vector to duplicate individual entries

# the simplest type of subsetting is "nothing"
# x[] will return the entire vector
# example in a matrix or data.frame = we can subset out specific rows and columns!!!
# x[1, ] selects the first row of all columns!!
# x[ ,1] selects all rows from the first column
# we can use double brackets [[]] to extract single items out of a list!!



## Recursive Vectors (Lists)
# lists are step up in complexity from atomic vectors
# lists can contain other lists!!
# this makes them suitable for representing hierachical or tree-like structures
x <- list(1,2,3)
x
# [[1]]
# [1] 1
# 
# [[2]]
# [1] 2
# 
# [[3]]
# [1] 3


# we view the structure of a list using str()
str(x)
# List of 3
# $ : num 1
# $ : num 2
# $ : num 3

x_named <- list(a = 1, b = 2, c = "fuck you bitch")
str(x_named)
# List of 3
# $ a: num 1
# $ b: num 2
# $ c: chr "fuck you bitch"

# unlike atomic vectors lists() can contain a mix of objects
y <- list("a", 1L, 1.5, T)
str(y)
# List of 4
# $ : chr "a"
# $ : int 1
# $ : num 1.5
# $ : logi TRUE

# lists can even contain other lists!!
z <- list(list(1,2), list(3,4))
str(z)
# List of 2
# $ :List of 2
# ..$ : num 1
# ..$ : num 2
# $ :List of 2
# ..$ : num 3
# ..$ : num 4


# Visualizing Lists
# to explain more complicated list manipulations functions we need to visualize lists
# example:
x1 <- list(c(1,2), c(3,4))
x2 <- list(list(1,2), list(3,4))
x3 <- list(1, list(2, list(3)))

# there are three main components:
        # lists have rounded corners (in the visual)
        # atomic vectors have square corners (in the visual)
        # children are draw inside thier parent
        # the orientation of the children (rows or columns) isn't important

# Subsetting Lists
# there are three ways to subset a list
a <- list(a = 1:3, b = "string", c = pi, d = list(-1,-5))

# single bracket extracts a sublist []
# the result will always be a list
str(a[1:2])
# List of 2
# $ a: int [1:3] 1 2 3
# $ b: chr "string"

# like vectors you can subset with a logical, integer or character vector

# double bracket extracts a single component from a list and removes a level or hierarchy
str(y[[1]])
# chr "a"

str(y[[4]])
# logi TRUE

# $ is a shorthand for extracting named elements of a list
# this is similiar to [[]] but we do not need quotations
a$a
# [1] 1 2 3

a[["a"]]
# [1] 1 2 3


# the distinction between [ and [[ is really important for lists
# [[ drills down into the list while [ returns a new smaller list
# remember the pepper shaker example
# x[1] is a pepper shaker with the first packet
# x[2] is a pepper shaker with the second packet
# x[1:2] is a pepper shaker with the first and second packet
# x[[1]] is only the first packet
# x[[1]][[1]] is the actual contents of the first packet!!!



# Attributes
# any vector can contain arbitrary additional metadata through its attributes
# attributes as named list of vectors that can be attached to any object
# you can get and set attributes with attr() or see them all with attributes()
x <- 1:10
attr(x, "greeting")
# NULL

attr(x, "gretting") <- "Hi"
attr(x, "farewell") <- "Bye"
attributes(x)
# $gretting
# [1] "Hi"
# 
# $farewell
# [1] "Bye"

# there are three very important attributes that are used to implement fundamental parts of R
# names are used to name the element of a vector
# dimensions (dims) make a vector behave like a matrix or an array
# class is used to implement the S3 object oriented system


# we have covered names and dimensions but not touched class
# class defines how a generic function will work on our data
# depending on the class of our data - the generic function will pick the right class to perform the calculation on 
# generic functions will behave differently depending on the class of data that they are looking at!
# this is called object-oriented programming!!

# generic function example:
as.Date
# function (x, ...) 
#         UseMethod("as.Date")
# <bytecode: 0x10bb8ea40>
#         <environment: namespace:base>

# the call to UseMethod means this function is a generic function
# this function will call a specific method ( a function) based on the class of the object the inputted
# you can list all the methods for a generic with methods()
methods("as.Date")
# [1] as.Date.character as.Date.date      as.Date.dates     as.Date.default   as.Date.factor   
# [6] as.Date.numeric   as.Date.POSIXct   as.Date.POSIXlt  
# see '?methods' for accessing help and source code

# depending on the class of the object we called on as.Date, the generic function will search through the list of methods and find the right one to apply
# if x is character - as.Date will call as.Date.character()
# we can see the specific implementation of a method with getS3method()
getS3method("as.Date", "default")
# function (x, ...) 
# {
#         if (inherits(x, "Date")) 
#                 return(x)
#         if (is.logical(x) && all(is.na(x))) 
#                 return(structure(as.numeric(x), class = "Date"))
#         stop(gettextf("do not know how to convert '%s' to class %s", 
#                       deparse(substitute(x)), dQuote("Date")), domain = NA)
# }
# <bytecode: 0x111137790>
#         <environment: namespace:base>

getS3method("as.Date", "numeric")
# function (x, origin, ...) 
# {
#         if (missing(origin)) 
#                 stop("'origin' must be supplied")
#         as.Date(origin, ...) + x
# }
# <bytecode: 0x1111091f8>
#         <environment: namespace:base>

# the most important S3 generic is print()
# it controls how the object is printed when you type its name at the console



## Augmented Vectors
# atomic vectors and lists are the building blocks for other important vector types like factors and dates
# these are augmented vectors because they are vectors with additional attributes including class
# because augmented vectors have a class they behave differently than the atomic vector that they are built on
# there are four important augmented vectors
        # factors
        # date-times and times
        # tibbles

# Factors
# factors are designed to represent categorical data than can take a fixed set of possible values
# factors are built on top of integers and have a levels attribute:
x <- factor(c("ab","cd","ab"), levels = c("ab","cd","ef"))
typeof(x)
# [1] "integer"

attributes(x)
# $levels
# [1] "ab" "cd" "ef"
# 
# $class
# [1] "factor"

# Dates and Date-Times
# dates in R are numeric vectors that represent the number of days since January 1 1970
x <- as.Date("1971-01-01")
unclass(x)
# [1] 365

typeof(x)
# [1] "double"
attributes(x)
# $class
# [1] "Date"

# date times are numeric vectors with class POSIXct that represent the number of seconds since 1970-01-01
# POSIXct stands for Portable Operating System Interface - aka calendar time
x <- lubridate::ymd_hm("1970-01-01 01:00")
unclass(x)
# [1] 3600
# attr(,"tzone")
# [1] "UTC"

typeof(x)
# [1] "double"
attributes(x)
# $tzone
# [1] "UTC"
# 
# $class
# [1] "POSIXct" "POSIXt"

# the tzone attribute is optional - it controls how the time is printed not what absolute time it refers to:
attr(x, "tzone") <- "US/Pacific"
x
# [1] "1969-12-31 17:00:00 PST"

attr(x, "tzone") <- "US/Eastern"
x
# [1] "1969-12-31 20:00:00 EST"

# there is another type of date-time called POSIXlt
# these are built on top of named lists
y <- as.POSIXlt(x)
typeof(y)
# [1] "list"
attributes(y)
# $names
# [1] "sec"    "min"    "hour"   "mday"   "mon"    "year"   "wday"   "yday"   "isdst"  "zone"   "gmtoff"
# 
# $class
# [1] "POSIXlt" "POSIXt" 
# 
# $tzone
# [1] "US/Eastern" "EST"        "EDT" 

# POSIXlts are rare in the tidyverse - they are needed to extract specific components of date - BUT WE HAVE LUBRIDATE


# Tibbles
# tibbles are augmented lists
# they have three classes: tbl_df, tbl, data.frame
# they have two attributes: column names and row.names
tb <- tibble::tibble(x = 1:5, y = 5:1)
typeof(tb)
# [1] "list"
attributes(tb)
# $names
# [1] "x" "y"
# 
# $row.names
# [1] 1 2 3 4 5
# 
# $class
# [1] "tbl_df"     "tbl"        "data.frame"

# traditional data frames have a very similiar structure
df <- data.frame(x = 1:5, y = 5:1)
typeof(df)
# [1] "list"
attributes(df)
# $names
# [1] "x" "y"
# 
# $row.names
# [1] 1 2 3 4 5
# 
# $class
# [1] "data.frame"

# the main difference between tibble and data frame is the class
# the class of tibble includes data.frame which means tibbles inherit the regular data.frame behavior by deafult
# the difference between a tibble or a dataframe and a list is that all of the elements of a tibble or df must be vectors with the same length (matrix style)













# chapter 17 iteration ----------------------------------------------------




## Chapter 17: Iteration with purrr
# we talked that writing functions to reduce code duplication is important
# reducing duplication has three main benefits:
        # its easier to see the intent of your code - eyes go to what is different
        # it is easier to respond to changes in requirements - changes are only made in one place
        # fewer bugs
# one tool for reducing duplication is functions
# we extract the process and put all duplicated work into a single function
# another tool for reducing duplication is iteration
# this helps you when you need to do the same thing to multiple inputs
# we repeat the same operation for every column or on different datasets
# there are two main iteration paradigms: imperative programming and functional programming
# imperative programming you have tools like for and while loops - iteration is very explicit
# loops are quite verbose and require a bit of bookkeeping code that is duplicated for each loop
# functional programming (FP) offers tools to extract out this duplicated code so each common loop pattern gets its own function
library(tidyverse)

# we will start with mastering loops in base R
# then move towards purrr to help with iteration

# start with a simple tibble
df = tibble(
        a = rnorm(10),
        b = rnorm(10),
        c = rnorm(10),
        d = rnorm(10)
)

# we want to compute the median of each column
# we could do this with copy and paste 
median(df$a)
# [1] -0.515092
median(df$b)
# [1] -0.1370709

# ...etc...


# this breaks our rule of thumb that we should never copy / paste more than twice
# we could use a for loop
output = vector("double", ncol(df)) # output

for (i in seq_along(df)) { # sequence
        output[[i]] = median(df[[i]]) # body
}

output # results
# [1] -0.5150920 -0.1370709 -0.2185346
# [4]  0.3013022


# every for loop has three components:

# output
        # before you start the loop we must always allocate space for the output
        # this is extremely important for efficiency
        # a general way of creating an empty vector of given length is the vector() function
        # it takes two arguements: the type of vector and the length of the vector

# sequence
        # this determines what to loop over:    
        # each run of the for loop will assign i to a different value from seq_along
        # seq_along is similiar to 1:length(l) however it will do the right thing when a zero length vector comes along

# body
        # this is the code that actually does the work
        # it is run repeatedly based on your definition in the sequence
        # each time run it will use a different value for i

# practice examples:

# For Loop Variants
# once you have the basic loop known there are some important variations
# there are four variations to the basic for loop theme:
        # modifying an existing object instead of creating a new object
        # looping over names or values instead of indices
        # handling outputs of unknown length
        # handling sequences of unknown length

# Modifying an existing object
# we can use a for loop to modify an existing object
# example we want to rescale every column in our data frame

# start with a simple tibble
df = tibble(
        a = rnorm(10),
        b = rnorm(10),
        c = rnorm(10),
        d = rnorm(10)
)

# build our rescale function
rescale01 = function(x) {
        rng = range(x, na.rm = T)
        (x - rng[1]) / (rng[2] - rng[1])
}

# we will still have to copy paste alot to rescale each column!
df$a = rescale01(df$a)
df$b = rescale01(df$b)

# to solve this as a loop we need to think of our three components
# output = we already have the output it is the same df as the input
# sequence = we can think about a data frame as a list of columns we can iterate over each column with seq_along(df)
# body = apply our rescale01 function!!!

for (i in seq_along(df)) {
        df[[i]] = rescale01(df[[i]])
}

df
# # A tibble: 10 x 4
#       a     b      c     d
#       <dbl> <dbl>  <dbl> <dbl>
# 1 0      0.519 0.674  0.143
# 2 0.420  0.800 0.707  0.477
# 3 0.535  0     0.607  0.396
# 4 0.313  0.453 0.146  0.306
# 5 0.606  0.499 0.544  1.00 
# 6 0.479  0.414 0.330  0.505
# 7 0.0934 1.00  1.00   0.922
# 8 1.00   0.698 0.0592 0.143
# 9 0.238  0.284 0.368  0.392
# 10 0.210  0.263 0      0 

# typically we will be modifying a list or data frame with this sort of loop so we need [[]]
# [[]] specifies a single element! might be good to use [[]] in all loops!

# Looping patterns
# there are three basic ways to loop over a vector
# looping over the numeric indicies and extracting the values in the body 
# there are two other forms:
        # loop over elements:
        # loop over names:
# if we are creating a named output make sure to name the results vector like below:
results = vector("list", length(x))
names(results) = names(x)

# iteration over the numeric indicies is the most general form
# given the position you can extract both the name AND the value
for (i in seq_along(x)) {
        name = names(x)[[i]]
        value = x[[i]]
}

name;value
# NULL
# [1] 100


# Unknown Output Length
# someimtes you might not know how long the output will be
# we could be simulating random vectors of random lengths
# we could be able to loop by progressively growing the vector but this is not efficient!!
means = c(0,1,2)

output = double()

for (i in  seq_along(means)) {
        n = sample(100,1)
        output = c(output, rnorm(n, means[[i]]))
}

str(output)
# num [1:234] 1.1692 0.0792 -0.4518 1.642 -0.7696 ..

# This is inefficient becuase R has to copy all the data from the previous iterations!
# this is quadratic (O(n^2)) behvaior which means a loop with 3 times an many elements would take nine (3^2) times as long to run

# a better solution is to save the results in a lost and then combine into a single vector after the loop is done
out = vector("list", length(means))

for (i in seq_along(means)) {
        n = sample(100,1)
        out[[i]] <- rnorm(n, means[[i]])
        }

out
# [[1]]
# [1]  1.248369640 -1.623245918 -0.002656146 -0.837264285  0.118853245 -0.501007298
# [7]  1.025344726  0.217808213  0.085578656  0.387769029 -1.817656629 -1.174550153
# [13]  1.366606514  0.190588625  0.407143173 -1.405579071  2.172244168 -0.539542776
# [19]  0.486887772 -0.261532000  0.923313589 -0.910911965  0.983827544  1.003069942
# [25] -0.790760348 -1.658033573  0.400994752 -2.348004883  0.730477590  0.755255204
# [31]  0.788566380 -0.342063446  1.823848892 -0.148140480 -0.971565897 -0.389138968
# [37] -1.263656166 -1.278265506  0.204386956  0.046567748 -0.908244134
# 
# [[2]]
# [1]  1.555737185  0.939880809  1.772086304  0.859160613  1.393093926  1.224218574
# [7]  1.023541985  0.377037340  2.262009381  0.594225957  1.666763771  1.164639155
# [13]  2.781524475  1.711213964  0.662308844  0.990851048  0.874690792 -1.090846097
# [19]  2.697393895  2.063881154  0.233383364  1.382007559  1.241895904 -0.132759411
# [25]  2.489907414  0.751752895  1.183583708  1.404871009  0.005875531 -0.085429330
# [31]  0.951457445  1.576085601  1.073830532  1.705945571  1.334980103  1.545387806
# [37] -0.402905906  1.677053891  0.210199554  0.534271111  0.895147935 -0.647851087
# 
# [[3]]
# [1]  0.30110054  1.61598056  1.19122687  1.47511063  2.71488719  2.70202929  1.23129987
# [8]  0.12287590  1.22947741  3.33396653  1.97970646  1.11802273  2.00872802  2.82282609
# [15]  3.86528990  2.54131035  1.00961239  4.27707702  2.08808937  3.42103578  3.30179821
# [22] -0.03791607  3.64086197  1.89007390  1.16204061  2.07467112  2.08131843  2.02569117
# [29]  1.85209925  3.50284367  3.03209248  2.33735470  1.41856575  3.24499489  2.69065315
# [36]  1.58291827  1.51018399  1.75771862  1.84692111  2.39039521  2.89018093  2.22508840
# [43]  0.32070836  2.36016953  3.10947973  0.69378510  0.11197667

str(unlist(out))
# num [1:130] 1.24837 -1.62325 -0.00266 -0.83726 0.11885 ...

# above we use unlist to flatten a list of vectors into a single vector
# a stricter option would be to use purrr::flatten_dbl() - it will give and error if no double available
# this pattern occurs in other places too
# you might be generating a long string
        # instead of paste'ing together each iteration - save the output in a character vector and then flatten with collapse
# you might be generating a big data frame
        # instead of rbinding each iteration - save the output in a list and then use dplyr::bind_rows to combine output into a single data frame

# watch out for this pattern - when you see it - switch to a more complex result object and then "flatten" or "add them up" at the end!!


## Unknown Sequence Length
# sometimes we do not know how long the input sequence will be 
# this is common in simulations
# we might want to stop a loop until we get three heads in a row
# to do this we will use a while loop
# a while loop only has two components a condition and a body

while (condition) {
        #body - do something here until condition is met
}

# a while loop is more general than a for loop because you can rewrite any for loop as a while loop
# we cannot rewrite every while loop as a for loop:
for (i in seq_along(x)) {
        # body - do something here
}

i = 1

# while loop equivalent
while (i <= length(x)) {
        # body - do something here
        i = i + 1
}

# while loop to stop after finding three heads in a row
flip = function() sample(c("T", "H"), 1)

flips = 0
nheads = 0

# while loop
while (nheads < 3) {
        if(flip() == "H") {
                nheads = nheads + 1
        } else {
                nheads = 0
        }
        flips = flips + 1
        
}

flips
# [1] 9

# while loops are mostly used for simulations
# they are essential for problems where the number of iterations is not known in advance

# build a list of functions
# multiple the disp variable
# create new labels for am variable
trans = list(
        # for each x multiply by a result
        disp = function(x) x * .0163871,
        # for each x convert to a factor with labels auto and manual
        am = function (x) {
                factor(x, labels = c("auto", "manual"))
        }
)

names(trans)
# [1] "disp" "am"  

# for each name in trans (disp, am) apply the function to the mtcars column with the same name
# output is the mtcars dataset with disp multiplied by our function (all observations)
# and the am column transformed to named labels 
for (var in names(trans)) {
        mtcars[[var]] = trans[[var]](mtcars[[var]])
}


## For Loops vs. Functionals
# for loops are not as important in R as they are in other languages
# R is A FUNCTIONAL PROGRAMMING LANGUAGE
# it is possible to wrap up loops in a function and call that function instead of using the for loop directly

# let's see why this is important

df <- tibble(
        a = rnorm(10),
        b = rnorm(10),
        c = rnorm(10),
        d = rnorm(10)
)

# we want to compute the mean of every column
# we could do this with a for loop
output <- vector("double", length(df))

# for loop taking means of each column
for (i in seq_along(df)) {
        output[[i]] <- mean(df[[i]])
}

output
# [1] -0.3519694  0.1257132 -0.1046795  0.2862835

# we can also put this into a function
col_mean <- function(df) {
        
        output <- vector("double", length(df))
        
        for (i in seq_along(df)) {
                output[[i]] <- mean(df[[i]])
        }
        
        
}

# what if we want to expand our mean function to other summary statistics?
# we would have to re-copy our mean function and create a new function for each statistic

# median
col_median <- function(df) {
        
        output <- vector("double", length(df))
        
        for (i in seq_along(df)) {
                output[[i]] <- median(df[[i]])
        }
        
        
}

# sd
col_sd <- function(df) {
        
        output <- vector("double", length(df))
        
        for (i in seq_along(df)) {
                output[[i]] <- sd(df[[i]])
        }
        
        
}


# we broke the copy and paste 3 times rule!!
# how can we generalize this function to run for any statistic that we would like?

# imagine a set of repeating functions
f1 <- function(x) abs(x - mean(x)) ^ 1
f2 <- function(x) abs(x - mean(x)) ^ 2
f3 <- function(x) abs(x - mean(x)) ^ 3

# there is a lot of duplication that we can generalize by adding an additional agruement to the function
# this will allow us to run our function for every version of i!
f <- function(x, i) abs(x - mean(x)) ^ i

# we can do the same for our summary stats functions!!
col_summary <- function(df, fun) {
        
        out <- vector("double", length(df))
        
        for (i in seq_along(df)) {
                out[i] <- fun(df[[i]])
        }
        
        out
}

# test out our generalized function
col_summary(df, median)
# [1] -0.5150920 -0.1370709 -0.2185346  0.3013022

col_summary(df, sd)
# [1] 1.1148517 1.1004320 1.1595039 0.9464544


# the idea of passing a function to another function is an extremely powerful idea
# this is what makes R a functional programming langugage
# the rest of this chapter will focus on the purrr package which aims to use mapping to eliminate the need for for loops

# the goal of using purrr functions is to allow breaking up of common list manipulations into independent peices
        # purrr can take care of generalizing our function to every element in a list
        # you can compose large peices together with piped operations in purrr
# the structure makes it easier to solve new complex problems
# we can also understand solutions easier


## The Map Functions
# the pattern of looping over a vector, doing something to each element and saving the results is so common
# purrr package provide functions that do this for you
# there is one function for each type of output
        # map() makes a list
        # map_lgl() makes a logical vector
        # map_int() makes an integer vector
        # map_dbl() makes a double vector
        # map_char() makes a character vector
# each function takes a vector as an input, applies a function to each peice and returns a new vector that's the same length as the input
# the type of output vector is determined by the function that we use

# once we master these we will be quicker at solving iteration problems
# for loops are still an option
# map functions are a step up in abstraction and take a longer time to understand how they work
# solve the problem you are working on

# for loops are no longer slow
# the main benefit of using map() functions is not speed but rather clarity
# we can use map functions to rewrite our previous for loop examples
# we specifiy the map function based on the data type we want our output vector to be in

# mean loop into doubles
map_dbl(df, mean)
# a          b          c          d 
# -0.3519694  0.1257132 -0.1046795  0.2862835 

# median loop into double
map_dbl(df, median)
# a          b          c          d 
# -0.5150920 -0.1370709 -0.2185346  0.3013022

# sd loop into double
map_dbl(df, sd)
# a         b         c         d 
# 1.1148517 1.1004320 1.1595039 0.9464544 

# compared to using a for loop the focus is on the operation performed
# not the setup required to loop over each element and store an output
# we can using the piping to make our code even more clear

# mean
df %>% map_dbl(mean)
# a          b          c          d 
# -0.3519694  0.1257132 -0.1046795  0.2862835 

# median
df %>% map_dbl(median)
# a          b          c          d 
# -0.5150920 -0.1370709 -0.2185346  0.3013022 

# sd
df %>% map_dbl(sd)
# a         b         c         d 
# 1.1148517 1.1004320 1.1595039 0.9464544 


# there are a few differences between map functions and our col_summary function
# purr functions are implemented in C which are faster
# the function arguement .f can take on many forms
# map uses ... to pass additional agruements to our mapped function

# mapped function with arguements
map_dbl(df, mean, trim = .5)
# a          b          c          d 
# -0.3509297  0.2976811  0.2858459 -0.2395249 

# mapped functions also retain our names
z <-  list(x = 1:3, y = 4:5)

map_int(z, length)
# x y 
# 3 2 


# shortcuts
# map functions have built in shortcuts that help save time
# suppose we want to fit a linear model multiple times to different splits of our dataset

# lm model fit for each factor of cylinder
models <- mtcars %>% 
        split(.$cyl) %>% 
        map(function(df) lm(mpg ~ wt, data = df))

# $`32`
# 
# Call:
#         lm(formula = mpg ~ wt, data = df)
# 
# Coefficients:
# (Intercept)           wt  
# 39.571       -5.647  
# 
# 
# $`48`
# 
# Call:
#         lm(formula = mpg ~ wt, data = df)
# 
# Coefficients:
# (Intercept)           wt  
# 28.41        -2.78  
# 
# 
# $`64`
# 
# Call:
#         lm(formula = mpg ~ wt, data = df)
# 
# Coefficients:
# (Intercept)           wt  
# 23.868       -2.192


# the lm call takes a long time to write
# map functions allows a one sided formula to make typing this easier

# map with shortcut - one sided formula
(models <- mtcars %>% 
        split(.$cyl) %>% 
        map(~lm(mpg ~ wt, data = .)))
# $`32`
# 
# Call:
# lm(formula = mpg ~ wt, data = .)
# 
# Coefficients:
# (Intercept)           wt  
# 39.571       -5.647  
# 
# 
# $`48`
# 
# Call:
# lm(formula = mpg ~ wt, data = .)
# 
# Coefficients:
# (Intercept)           wt  
# 28.41        -2.78  
# 
# 
# $`64`
# 
# Call:
# lm(formula = mpg ~ wt, data = .)
# 
# Coefficients:
# (Intercept)           wt  
# 23.868       -2.192 


# when we are looking at so many models we might want to extract a summary statistic for each model
# we could do this using the summary function
models %>% 
        map(summary) %>% # map summary to each lm object in models
        map_dbl(~.$r.squared) # map the extract of r squared from each summary of each lm model
# 32        48        64 
# 0.5086326 0.4645102 0.4229655 

# extracting named components is a common operation
# purrr provides an even shorter shortcut - we can use a string to extract from our models
models %>% 
        map(summary) %>% 
        map_dbl("r.squared")
# 32        48        64 
# 0.5086326 0.4645102 0.4229655

# we can also use an integer to select elements by position:
x <- list(list(1,2,3), list(4,5,6), list(7,8,9))

# extract the second element for each element within x
x %>% map_dbl(2)
# [1] 2 5 8


## Base R
# if you are familiar with the apply functions in base R there are some simliarities with map
# lapply is basically identical to map but we can use short cuts for our mapped function .f
# base sapply() is a wrapper around lapply() that automatically simplies the output
# vapply is a safe alternative to sapply becuase we define what the output will look like

# purrr has more consistent names and arguements, helpful shortcuts



## Dealing with Failure
# when we use the map functions to repeat many operations, the chances are one of those operations will eventually fail
# we will get an error message when this happens and no output
# why can't we still get all the other successes?
# we have a new function safely() that helps with this operations
# this will return a list with two elements:
        # result: the original result: if there is an error this will be NULL
        # error: an error object: if the operation was successful this will be NULL
# this is similiar to try() but will always return two list elements instead of just one in try()

# example
safe_log <- safely(log)

str(safe_log(10))
# List of 2
# $ result: num 2.3
# $ error : NULL

str(safe_log("a"))
# List of 2
# $ result: NULL
# $ error :List of 2
# ..$ message: chr "non-numeric argument to mathematical function"
# ..$ call   : language log(x = x, base = base)
# ..- attr(*, "class")= chr [1:3] "simpleError" "error" "condition"

# when the formula succeeds the result element contains the result and the error element in NULL
# when the function fails, the result element is NULL and the error element contains an error object

# safely() is designed to work with map()
x <- list(1, 10, "a")
y <- x %>% 
        map(safely(log))
str(y)
# List of 3
# $ :List of 2
# ..$ result: num 0
# ..$ error : NULL
# $ :List of 2
# ..$ result: num 2.3
# ..$ error : NULL
# $ :List of 2
# ..$ result: NULL
# ..$ error :List of 2
# .. ..$ message: chr "non-numeric argument to mathematical function"
# .. ..$ call   : language log(x = x, base = base)
# .. ..- attr(*, "class")= chr [1:3] "simpleError" "error" "condition"

# this would be easier to work with if we had two lists:
# one of all the errors and one of all the output
# we can use purrr::transpose() to do this
y <- y %>% transpose()
str(y)
# List of 2
# $ result:List of 3
# ..$ : num 0
# ..$ : num 2.3
# ..$ : NULL
# $ error :List of 3
# ..$ : NULL
# ..$ : NULL
# ..$ :List of 2
# .. ..$ message: chr "non-numeric argument to mathematical function"
# .. ..$ call   : language log(x = x, base = base)
# .. ..- attr(*, "class")= chr [1:3] "simpleError" "error" "condition"


# we can either look at the values of x where y is an error or work with the values that go through correctly
(is_ok <- y$error %>% map_lgl(is_null))
# [1]  TRUE  TRUE FALSE
x[!is_ok]
# [[1]]
# [1] "a"

y$result[is_ok] %>% flatten_dbl()
# [1] 0.000000 2.302585

# purrr provides two other useful adverbs
# possibly() always succeeds - you also give it a default value to throw when an error occurs

x <- list(1, 10, "a")
x %>% map_dbl(possibly(log, NA_real_))
# [1] 0.000000 2.302585       NA

# quietly captures printed output, messages and warnings
x <- list(1, -1)
x %>% map(quietly(log)) %>% str()
# List of 2
# $ :List of 4
# ..$ result  : num 0
# ..$ output  : chr ""
# ..$ warnings: chr(0) 
# ..$ messages: chr(0) 
# $ :List of 4
# ..$ result  : num NaN
# ..$ output  : chr ""
# ..$ warnings: chr "NaNs produced"
# ..$ messages: chr(0) 


## Mapping over Multiple Arguements
# so far we've mapped along a single input
# we often will have multiple related inputs that we need to iterate along in parrellel
# we will use the map2() and pmap functions to do this

# imagine we want to simulate random normals with different means
# we can do this with our normal map()

mu = list(5, 10, -3)
mu %>% 
        map(rnorm, n = 5) %>% 
        str()
# List of 3
# $ : num [1:5] 5.73 4.12 3.46 3.96 3.28
# $ : num [1:5] 10.8 8.5 9.85 10.58 11.2
# $ : num [1:5] -1.11 -4.76 -2.08 -3.56 -3.18

# what if we also want to vary the sd?
# one way would be to iterate over the indices and index into vectors of means and sd

sigma = list(1, 5, 10)
seq_along(mu) %>% 
        map(~rnorm(5, mu[[.]], sigma[[.]])) %>% 
        str()
# List of 3
# $ : num [1:5] 6.45 4.39 5.68 4.91 4.51
# $ : num [1:5] 17.05 8.88 8.94 13.48 14.58
# $ : num [1:5] -12.23 8.47 -9.36 -11.86 -26.33

# this makes our code hard to read...
# instead we could use map2() with iterates over two vectors in parallel
map2(mu, sigma, rnorm, n = 5) %>% str()
# List of 3
# $ : num [1:5] 4.85 5.32 4.29 6.24 5.62
# $ : num [1:5] 10.5 19.04 2.49 11.43 14.23
# $ : num [1:5] -12.95 -5.57 -3.56 -7.45 -2.3

# map2 generates:
        # mu vector 5, 10, -3
        # sigma vector 1, 5, 10
        # rnorm(5, 1, n = 5)
        # rnorm(10, 5, n = 5)
        # rnorm(-3, 10, n = 10)

# note that the arguements that vary for each call come before the function
# arguements that are the same for every call come after
# like map, map2 is just a wrapper around a for loop:
map2

function (.x, .y, .f, ...) 
{
        .f <- as_mapper(.f, ...)
        .Call(map2_impl, environment(), ".x", ".y", ".f", "list")
}

# map2 is a wrapper around a for loop
map2 = function(x, y, f, ...) {
        out = vector("list", length(x))
        
        for (i in seq_along(x)) {
                out[[i]] = f(x[[i]], y[[i]], ...)
        }
        
        out
}



# we would imagine map3 and up but that would get tedious quickly
# we will use pmap instead
# we could use this if we wanted to vary the mean, standard deviation and number of samples
n = list(1,3,5)
args1 = list(n, mu, sigma)
args1 %>% 
        pmap(rnorm) %>% 
        str()
# List of 3
# $ : num 4.85
# $ : num [1:3] 5.84 13.81 7.12
# $ : num [1:5] -9.26 1.81 13.95 -20.61 -1.02


# this will look like:
# rnrom(1,5,1)
# rnorm(3,10,5)
# rnorm(5, -3, 10)

# if we do not name the elements of list, pmap wil use poositional matching when calling the function
# it is better to name the arguements
args2 = list(mean = mu, sd = sigma, n = n)
args2 %>% 
        pmap(rnorm) %>% 
        str()
# List of 3
# $ : num 5.4
# $ : num [1:3] 10.1 22.8 16.3
# $ : num [1:5] -8.35 -9.25 6.14 7.07 4.19


# this generates longer but safer calls
# rnorm(mu = 5, sigma = 1, n = 1)
# rnorm(mu = 10, sigma = 5, n = 3)
# nrorm(mu = -3, sigma = 10, n = 5)

# since the arguements are all the same length it makes sense to store them in a data frame
params = tribble(
        ~mean, ~sd, ~n,
        5,1,1,
        10,5,3,
        -3,10,5
)

params %>% 
        pmap(rnorm) %>% 
        str()
# List of 3
# $ : num 6.32
# $ : num [1:3] 5.09 8.77 2.98
# $ : num [1:5] 11.41 -12.81 11.74 -12.91 -3.94

# as soon as our code gets complicated a data frame is a good approach
# it ensures each column has a name and is the same length as all other columns


## Invoking different Functions
# there is one more step up in complexity of our map functions
# we might want to vary the function itself
f = c("runif", "rnorm", "rpois")
param = list(
        list(min = -1, max = 1),
        list(sd = 5),
        list(lambda = 10)
)

# to handle this we can use invoke_map
invoke_map(f, param, n = 5) %>%  str()
# List of 3
# $ : num [1:5] -0.996 0.2384 -0.195 0.8001 0.0118
# $ : num [1:5] 3.01 -1.54 -2.09 1.78 2.57
# $ : int [1:5] 10 14 9 7 12


# here is what this call looks like:
# f = runif, rnorm, rpois
# params = -1,1 for runif; sd = 5 for rnorm, lambda = 10 for rpois
# final calls:
# runif(min = -1, max = 1, n = 5)
# rnorm(sd = 5, n = 5)
# rpois(lambda = 10, n = 5)

# our first arguement is a list of functions or a character vector of function names
# the second arguement is a list of lists giving the agruments that vary for each function
# the last arguements (n = 5) is passed to every function within the function list

# we can use tribble to make creating these pairs a little easier
sim = tribble(
        ~f, ~params,
        "runif", list(min = -1, max = 1),
        "rnorm", list(sd = 5),
        "rpois", list(lambda = 10)
)

sim %>% 
        mutate(sim = invoke_map(f, params, n = 10))
# # A tibble: 3 x 3
# f     params     sim       
# <chr> <list>     <list>    
#         1 runif <list [2]> <dbl [10]>
#         2 rnorm <list [1]> <dbl [10]>
#         3 rpois <list [1]> <int [10]>


## Walk
# walk is an alternative to map that you use when you want to call a funciton for its side effects, rather than return a value
# we do this when we want to render something to the screen or save files to disk
# the important thing is the action not a returned value

# simple example:
x = list(1, "a", 3)

x %>% walk(print)
#[1] 1
# [1] "a"
# [1] 3


# we also have walk2 and pwalk functions that we can use
# if we had a list of plots and a list of files names we could use pwalk to save them all to disk
library(ggplot2)

# create three plots 
plots = mtcars %>% 
        split(.$cyl) %>% 
        map(~ggplot(., aes(mpg, wt)) + geom_point())

# define paths to save
paths = stringr::str_c(names(plots), ".pdf")

# save each plot with specific name
pwalk(list(paths, plots), ggsave, path = tempdir())

# all walk statements invsibily reutrn .x the first arguement - this makes them still useful in piping!!



## Other Patterns of For Loops
# purrr provides additional functions to abstract over for loops


## Predicate functions:
# a number of functions work with predicate functions that reutrn either a single T or F

# keep and discard
# kepp elements of the input where the predicate is T or F
iris %>% 
        keep(is.factor) %>% 
        str()
# 'data.frame':	150 obs. of  1 variable:
# $ Species: Factor w/ 3 levels "setosa","versicolor",..: 1 1 1 1 1 1 1 1 1 1 ...

iris %>% 
        discard(is.factor) %>% 
        str()
# 'data.frame':	150 obs. of  4 variables:
# $ Sepal.Length: num  5.1 4.9 4.7 4.6 5 5.4 4.6 5 4.4 4.9 ...
# $ Sepal.Width : num  3.5 3 3.2 3.1 3.6 3.9 3.4 3.4 2.9 3.1 ...
# $ Petal.Length: num  1.4 1.4 1.3 1.5 1.4 1.7 1.4 1.5 1.4 1.5 ...
# $ Petal.Width : num  0.2 0.2 0.2 0.2 0.2 0.4 0.3 0.2 0.2 0.1 ...


# some and every
# determines if the predicate is true for any or all of the elements
x = list(1:5, letters, list(10))

x %>% some(is.character)
# [1] TRUE

x %>% every(is.vector)
# [1] TRUE


# detect
# finds the first element where the predicate is T
# detect index returns its position
x = sample(10)
x
# [1]  5  7  2  6  1  8 10  3  4  9

# FINDS THE FIRST ELEMENT WHERE OUR PREDICATE STATEMENT IS TRUE!
x %>% detect(~ . > 5)
# [1] 7

x %>% detect_index(~. > 5)
# [1] 2


# head_while, tail_while
# take the elements from the start or end of a vector while the predicate is true
x %>% head_while(~. > 5)
# [1] 6


x %>% tail_while(~ . > 5)
# [1] 10


## Reduce and Accumluate
# someimtes you have a complex list that you want to reduce to a simple list
# we can do this by repeatedly applying functions that reduce a pair to a single
# this is useful if you want to apply a two-table dpylr verb to multiple tables

# example: list of data frames we want to join together
dfs <- list(
        age <- tibble(name = "John", age = 30), 
        sex <- tibble(name = c("John", "Mary"), sex = c("M", "F")),
        trt <- tibble(name = "Mary", treatment = "A")
)

# reduce list of dataframes into one iteratively using full join
# will join df1 and df2, then move onto the joined df1 and df2 to df3
# repeats until we are left with a single entity
dfs %>% reduce(full_join)
# Joining, by = "name"
# Joining, by = "name"
# # A tibble: 2 x 4
# name    age sex   treatment
# <chr> <dbl> <chr> <chr>    
# 1 John   30.0 M     NA       
# 2 Mary   NA   F     A  

# reduce a list of vectors by thier intersection
vs <- list(
        c(1,2,3),
        c(1,1.5,3),
        c(1,3,10)
)

# find itersection of list item 1 and list item 2, 
# then move onto intersect of list item 3 and the results from list item 1 and item 2
# continue until we are left with one list element that shows the interection between all three original list elements
vs %>% reduce(intersect)
# [1] 1 3




# chapter 18 model basics -------------------------------------------------

# now that we have the powerful programming tools we can finally get into modeling
# we will use our new data tools of data wrangling to fit models onto our tidy data
# the goal of a model is to provide a simple low-dimensional summary of a dataset
# the model should capture the true signals of the underlying dataset and ignore the noise
# there are many different types of models from prediction to data discovery
# chapter will focus on giving you intuition on how models work
# chapter 18: how models work mechanically
# Chapter 19: use models to pull out unknown patterns in real data
# Chapter 20: learn how to use many simple models to help understand complex data sets

# hypothesis generation vs. hypothesis confirmation
# we are going to use models as a means of exploration in this book
# inference will not be covered
# doing models correctly is not complicated but it is hard

# inference ideas / concepts:
# each observation can either be used for exploration or confirmation - not both (test and train)
# you can use an observation as many times for exploration but only once for confirmation

# to confirm our hypothesis we need to validate it on independent data 
# we need to confirm our exploratory models by testing it against data the models have not seen

# training: 60% of data does into training set - fit our models on this data
# query: use this data to compare models this is also known as validation
# test: the held back test set - use this data once to test our final model

# this partitioning allows you to explore and test our best models


## Chapter 18: Model Basics with modelr
# the goal of a model is to provide a simple low-dimensional summary of a dataset
# we are going to use models to partition data into patterns and residuals
# before we get started we will need to understand how models actually work
# we will apply our model building to simulated data first to understand the functions of modeling in R

# there are two parts of a mdeol:
# define the family of models that express a precise but generic pattern we want to capture
# generate a fitted model by finding the model from the family closest to your data

# it is important to understand that a fitted model is just the closest model from a family of models
# we can determine the "best" model for our  underlying data set but it doesn't mean that it is good
# all models are wrong but some are useful
# the goals of models is not to uncover the absolute truth but to discover approximations that are useful

# load tidyverse and modelr
library(tidyverse); library(modelr)

## a simple model
# lets take a look at dataset sim1
# it contains two continous variables x and y - we want to see how they are related to each other

ggplot(sim1, aes(x, y)) + 
        geom_point()

# we can see a strong pattern in the data - a strong positive correlation
# let's use a model to capture that pattern and make it explicit
# we supply the basic form of the model and R will fit the model as best it can

# define models
models = tibble(
        a1 = runif(250, - 20, 40),
        a2 = runif(250, -5, 5)
)

# manually fit using geom_abline
ggplot(sim1, aes(x, y)) +
        geom_abline(
                aes(intercept = a1, slope = a2),
                data = models, alpha = 1/4
        ) +
        geom_point()

# there are over 250 models in this plot and a lot are really bad at finding the true relationship
# we want to find a line that fits the relationship of this data as close as possible
# we do this my minimizing the error from the actual data points to our line
# this distance is just the difference between the y value given by model (prediction) and the actual observation (actuals / repsonse)

# to compute this distance we first turn our model family into an R function
# this takes the model parameters and the data as inputs and gives values predicted by the model as output:
model1 = function(a, data) {
        a[1] + data$x * a[2]
}

# use the function
model1(c(7, 1.5), sim1)
# [1]  8.5  8.5  8.5 10.0 10.0 10.0 11.5 11.5 11.5 13.0
# [11] 13.0 13.0 14.5 14.5 14.5 16.0 16.0 16.0 17.5 17.5
# [21] 17.5 19.0 19.0 19.0 20.5 20.5 20.5 22.0 22.0 22.0

# next we need some way to compute an overall distance between the predicted and actual values
# we can do this with the root mean sd
# we compute the difference between the actual and predicted, square them, average them, and then take square root

# root mean squared error function
measureDistance = function(mod, data) {
        diff = data$y - model1(mod, data) # actuals - predicted
        sqrt(mean(diff^2)) # RMSE calculation
}

# calculate distance from actuals to fitted line
measureDistance(c(7, 1.5), sim1)
# [1] 2.665212


# we can use purrr to compute the distance for all the models defined previously - all 250 models!
# we need a helper function because our distance function expects the model as a numeric with vector length of 2

# helper function
sim1_dist = function(a1, a2) {
        measureDistance(c(a1, a2), sim1)
}

# calculate distrance for all 250 models we tried to fit to our data
(models = models %>% 
        mutate(dist = map2_dbl(a1, a2, sim1_dist)))

# # A tibble: 250 x 3
# a1      a2  dist
# <dbl>   <dbl> <dbl>
# 1   9.53  1.74    4.29
# 2   4.76  4.03   12.9 
# 3 - 4.56  0.429  18.4 
# 4 -10.1  -4.00   50.8 
# 5  32.3   2.84   32.6 
# 6  17.9  -3.25   21.8 
# 7  37.8   1.32   29.7 
# 8  36.6   1.43   29.1 
# 9  37.3  -3.43   16.2 
# 10  33.9  -0.0411 19.2 
# # ... with 240 more rows

# next we want to overlay the 10 best models with the lowest distance from our calculation
# plot models in ggplot
ggplot(sim1, aes(x, y)) + 
        geom_point(size = 2, color = "grey30") +
        geom_abline(
                aes(intercept = a1, slope = a2, color = -dist), # define what our line is using slope and intercept
                data = filter(models, rank(dist) <= 10) # filter our models to the 10 best models
        )

# we can also think about these models as observations
# we can visualize them with a scatterplot of a1 vs. a2
# our best model will be the point where a1 and a2 both equal 0!!
ggplot(models, aes(a1, a2)) + 
        geom_point(
                data = filter(models, rank(dist) <= 10),
                size = 4, color = "red"
        ) + 
        geom_point(aes(color = -dist))


# instead of trying lots of random models we could be more systematic and generate a grid of points (grid search)
# grid search example

grid = expand.grid(
        a1 = seq(-5, 20, length = 25),
        a2 = seq(1,3, length = 25)
) %>% 
        mutate(dist = map2_dbl(a1, a2, sim1_dist))

# plot grid with top models
grid %>% 
        ggplot(aes(a1, a2)) + 
        geom_point(
                data = filter(grid, rank(dist) <= 10),
                size = 4, color = "red"
        ) + 
        geom_point(aes(color = -dist))

# when we overlay our ten best models back onto the original data they all look pretty good
ggplot(sim1, aes(x, y)) + 
        geom_point(size = 2, color = "grey30") + 
        geom_abline(
                aes(intercept = a1, slope = a2, color = -dist),
                data = filter(grid, rank(dist) <= 10)
        )

# we could imagine iteratively making the grid finer and finer until we select the best model
# the best model will minimize the distance between the predicted and actuals
# there is a better way to do this: Newton-Ralphson search
# pick random point > find steepest slope > go down that slope a little bit > repeat the process until we can't go any lower

# we can use the NR law with optim() in R
# this will give us the slope and intercept that minimizes the distance between predicted and actuals!!
best = optim(c(0,0), measureDistance, data = sim1)
best$par
# [1] 4.222248 2.051204

# if we have a function that defines the distance between a model and a dataset
# and an algorithm that can minimize the distance by modifying the parameters of the model
# WE CAN FIND THE BEST MODEL!!
# this approach will work for any model you can generate an equation for!

# linear models:
# linear models have the form: y = a_1 + a_2*x_1 + a_3*x_2 + ... + a_n*x_(n-1)
# we can use lm() to fit linear models
# we fit the model using the lm "formula" notation
sim1_mod = lm(y~x, data = sim1)
coef(sim1_mod)
# (Intercept)           x 
# 4.220822    2.051533

# these are exactly the same values we got with our optim() call!
# lm takes advantage of the mathematical structure of linear models
# lm finds the best line using a sophisticated algorithm
# this approach is faster and guarantees there is a global minimum

best <- optim(c(0,0), measureDistance, data = sim1)
best$par
# [1] 4.222248 2.051204

## Visualizing Models
# for simple models you can figure out what pattern the model captures by carefully studying the model family and the coeffiicents
# we are going to focus on understanding a model by looking at its predictions
# this has a big advantage:
# every type of predictive model makes predictions so we can use the same set of techniques on any model

# it is useful to see what our model does not capture - the residuals that are left after subtracting the predictions from the data
# residuals are powerful because they allow for us to use models to remove striking patterns so we can study the subtler trends that remain

## Predictions
# to visualize the predictions from a model we start by generating an evenly spaced grid of values that covers the region where our data lies
# the easiest way to do this is use modelr::data_grid()
# its first arguement is a data frame...
# and for each subsequent arguement it finds the unique variables and then generates all the combinations
(grid <- sim1 %>% 
        data_grid(x))
# # A tibble: 10 x 1
# x
# <int>
#         1     1
# 2     2
# 3     3
# 4     4
# 5     5
# 6     6
# 7     7
# 8     8
# 9     9
# 10    10

# next we add our predictions
# we'll use modelr::add_predictions()
# this takes a data frame and a model
# it adds the predictions from the model to a new column in the data frame
(grid <- grid %>% 
        add_predictions(sim1_mod))

# # A tibble: 10 x 2
# x  pred
# <int> <dbl>
# 1     1  6.27
# 2     2  8.32
# 3     3 10.4 
# 4     4 12.4 
# 5     5 14.5 
# 6     6 16.5 
# 7     7 18.6 
# 8     8 20.6 
# 9     9 22.7 
# 10    10 24.7 

# we can also use this function to add predictions to your original dataset

# next we plot the predictions
# this approach will work with any model in R from the most simple to the most complex
ggplot(sim1, aes(x)) +
        geom_point(aes(y = y)) +
        geom_line(
                aes(y = pred), 
                data = grid, 
                color = 'red',
                size = 1
        )

## Resiudals
# the flip side of predictions are reisudals
# the predictions tell you the pattern that the model has captured - residuals tell you what the model has missed!!
# the resiudals are just the distances between the observed and predicted values 

# we add resiudals to the data with add_residuals()
# we feed the original dataset because we need the true observed y values
(sim1 <- sim1 %>% 
        add_residuals(sim1_mod))
# # A tibble: 30 x 3
# x     y    resid
# <int> <dbl>    <dbl>
# 1     1  4.20 -2.07   
# 2     1  7.51  1.24   
# 3     1  2.13 -4.15   
# 4     2  8.99  0.665  
# 5     2 10.2   1.92   
# 6     2 11.3   2.97   
# 7     3  7.36 -3.02   
# 8     3 10.5   0.130  
# 9     3 10.5   0.136  
# 10     4 12.4   0.00763
# # ... with 20 more rows

# there are a few different ways to understand what the resiudals tell us about the model
# one way is to simply draw a frequency polygon to help us understand the spread of residuals
ggplot(sim1, aes(resid)) + 
        geom_freqpoly(binwidth = .5)

# this helps us understand the quality of the model:
# how far away are the predictions from the observed values
# the average residual will always be zero
# here we see our residuals range from -3 to 4

# we will often want to re-create plots using the resiudals isntead of the original predictor
ggplot(sim1, aes(x, resid)) + 
        geom_ref_line(h = 0) +
        geom_point()

# this plot is "patternless" meaning our residuals are randomly spreadout throughout the predictor x
# this means our model has done a good job of picking up on the true pattern in our dataset

## Formulas and Model Families
# formulas are a way of generating "special behavior"
# rather than evaluating the values of the variables right away
# they capture them so they can be interpreted by the function

# the majority of modeling functions use the standard convention formulas to functions
# y ~ x is translated to: y = a_1 + a_2 * x
# we can see what R is actually doing by using the model_matrix function
# it takes a dataframe and a formula and returns a tibbles that defines the model equation
# each column in the output is associated with one coefficient in the model...
# and the function is always y = a_1 * out1 + a_2*out2
# for the simplest case of y ~ x1 this shows us something interesting
df <- tribble(
        ~y, ~x1, ~x2,
        4,2,5,
        5,1,6
)

# model matrix function
model_matrix(df, y ~ x1)
# # A tibble: 2 x 2
# `(Intercept)`    x1
# <dbl> <dbl>
# 1          1.00  2.00
# 2          1.00  1.00

# the way R adds the intercept to the model is just by having a column full of ones
# we can explicitly drop this with -1
model_matrix(df, y ~ x1 - 1)
# A tibble: 2 x 1
# x1
# <dbl>
# 1  2.00
# 2  1.00

# the model matrix works the same way when you add in more variables!!
model_matrix(df, y ~ x1 + x2 - 1)
# # A tibble: 2 x 2
# x1    x2
# <dbl> <dbl>
# 1  2.00  5.00
# 2  1.00  6.00

# this formula notation is sometimes called the 'Wilkinson-Rogers' notation
# the next sections expand on how this formula notation works for categorical variables, interactions, and transformations

## Categorical Variables
# generating a function from a formula is straightforward when the predictor is continous but thing get complicated when a predictor is categorical
# imange a caegorical variable sex - which can be Male or Female
# sex isn't a number and cannot be pushed into the regular formula call
# R converts this variable to a dummy variable where male is 1 and female is 0
# this allows us to push in the categorical variable into the model formula
(df <- tribble(
        ~sex, ~response,
        'male', 1, 
        'female', 2,
        'male', 1
 ) %>% 
        model_matrix(., response ~ sex))
# # A tibble: 3 x 2
# `(Intercept)` sexmale
# <dbl>   <dbl>
# 1          1.00    1.00
# 2          1.00    0   
# 3          1.00    1.00

# R doesn't create a sexfemale column becuase then we would have a column that is perfectly predictable based on another column!!
# sexfemale = 1 - sexmale
# this creates a model family that is too flexible and will have infinitely many models that are equally close to our data

# if you focus on visualizing predictions you don't need to worry about the exact parameterization of the categorical variable
# let's look at some data and models to make this point stick
# we will use the sim2 dataset from modelr
ggplot(sim2) + 
        geom_point(aes(x, y))

# we can fit a model to this data and generate predictions
mod2 <- lm(y ~ x, data = sim2)

# generate predictions
(grid <- sim2 %>% 
        data_grid(x) %>% 
        add_predictions(mod2))
# # A tibble: 4 x 2
# x      pred
# <chr> <dbl>
# 1 a      1.15
# 2 b      8.12
# 3 c      6.13
# 4 d      1.91

# effectively a model with categorical x will predict the mean value for each category
# the mean minimizes the root mean squared distance!
# this is easy to see if we overlay the predictions on top of the original data:
ggplot(sim2, aes(x)) + 
        geom_point(aes(y = y)) + 
        geom_point(
                data = grid, 
                aes( y=pred), 
                color = 'red',
                size = 4
        )
# you can't make predictions about levels that you didn't observe
# sometimes you'll do this by accident so it's good to recognize this error message
tibble(x = 'e') %>% 
        add_predictions(mod2)
# Error in model.frame.default(Terms, newdata, na.action = na.action, xlev = object$xlevels) : 
#         factor x has new level e


## Interactions: Continous and Categorical
# what happens when you combine a continous and a categorical variable?
# sim3 contains a categorical predictor and a continous predictor

# visualize sim3
ggplot(sim3, aes(x1, y)) + 
        geom_point(aes(color = x2))

# there are two possible models that we can fit to this data
mod1 <- lm(y ~ x1 + x2, data = sim3)
mod2 <- lm(y ~ x1 * x2, data = sim3)

# when we add variables with the + the model will estimate each effect independent of all the others
# it's possible to fit the iteractions between each variable with the *
# for example y ~ x1 * x2 is translated to:
# y = a_0 + a_1 * a1 + a_2 * a2 + a_12 * a1 * a2
# note that whenever we use * we also include the interaction terms and the individual components in the model

# to visualize these models we need new tricks:
# we have two predictors so we need to give data_grid() both variables
# data_grid() finds all unqiue values of x1 and x2 and then generates all combinations
# to generate predictions of both models at the same time - we can use gather_predictions()
# this adds each prediction as a row
# the complement of gather_predictions() is spread_predictions() - this adds predictions as a new column

# together this gives us:
(grid <- sim3 %>% 
        data_grid(x1, x2) %>% 
        gather_predictions(mod1, mod2)
)
# # A tibble: 80 x 4
# model    x1 x2     pred
# <chr> <int> <fct> <dbl>
# 1 mod1      1 a      1.67
# 2 mod1      1 b      4.56
# 3 mod1      1 c      6.48
# 4 mod1      1 d      4.03
# 5 mod1      2 a      1.48
# 6 mod1      2 b      4.37
# 7 mod1      2 c      6.28
# 8 mod1      2 d      3.84
# 9 mod1      3 a      1.28
# 10 mod1      3 b      4.17
# # ... with 70 more rows

# we can visualize the results for both models on one plot using faceting:
ggplot(sim3, aes(x1,y, color = x2)) + 
        geom_point() + 
        geom_line(data = grid, aes(y = pred)) + 
        facet_wrap(~model)

# note that the model that uses + has the same slope for each line but different intercepts
# the interaction model that uses * has a different slope and intercept for each line

# which model is better for this data?
# we can take a look at the resiudals
# below we facet by both model and x2 to see the pattern within each group:
(sim3 <- sim3 %>% 
        gather_residuals(mod1, mod2)
)

# # A tibble: 480 x 8
# model model    x1 x2      rep      y    sd   resid
# <chr> <chr> <int> <fct> <int>  <dbl> <dbl>   <dbl>
# 1 mod1  mod1      1 a         1 -0.571  2.00 -2.25  
# 2 mod1  mod1      1 a         2  1.18   2.00 -0.491 
# 3 mod1  mod1      1 a         3  2.24   2.00  0.562 
# 4 mod1  mod1      1 b         1  7.44   2.00  2.87  
# 5 mod1  mod1      1 b         2  8.52   2.00  3.96  
# 6 mod1  mod1      1 b         3  7.72   2.00  3.16  
# 7 mod1  mod1      1 c         1  6.51   2.00  0.0261
# 8 mod1  mod1      1 c         2  5.79   2.00 -0.691 
# 9 mod1  mod1      1 c         3  6.07   2.00 -0.408 
# 10 mod1  mod1      1 d         1  2.11   2.00 -1.92  
# # ... with 470 more rows


# lets visualize the results across model and our categorical variables x2
# ideally we would want to see a residual plot with no clear pattern!!
ggplot(sim3, aes(x1, resid, color = x2)) +
        geom_point() + 
        facet_grid(model ~ x2)

# there is little obvious pattern in the resiudals for mod2
# the residuals for mod1 show that the model has clearly missed some pattern in b, and some in c and d
# there is a statistical way to quantify this difference - but we can use visual inspection for a quick check on the results
# here we are interested in the qualitative assessment of whether or not the model has captured the pattern that we are interested in

## Interactions Two Continous Variables
# lets take a look at the equivalent model for two continous variables

# continous interactions
mod1 <- lm(y ~ x1 + x2, data = sim4)
mod2 <- lm(y ~ x1 * x2, data = sim4)

# define grid and gather the predictions from each model
(grid <- sim4 %>% 
        data_grid(
                x1 = seq_range(x1, 5),
                x2 = seq_range(x2, 5)
        ) %>% 
        gather_predictions(mod1, mod2)
)

# # A tibble: 50 x 4
# model     x1     x2   pred
# <chr>  <dbl>  <dbl>  <dbl>
# 1 mod1  -1.00  -1.00   0.996
# 2 mod1  -1.00  -0.500 -0.395
# 3 mod1  -1.00   0     -1.79 
# 4 mod1  -1.00   0.500 -3.18 
# 5 mod1  -1.00   1.00  -4.57 
# 6 mod1  -0.500 -1.00   1.91 
# 7 mod1  -0.500 -0.500  0.516
# 8 mod1  -0.500  0     -0.875
# 9 mod1  -0.500  0.500 -2.27 
# 10 mod1  -0.500  1.00  -3.66 
# # ... with 40 more rows

# the seq_range() function subsets a range of possible x values so we do not have to build a grid for every value
# there are some other useful functions available in seq_range()

# pretty
# will generate a pretty sequence 
# nice for producing table output
seq_range(c(0.123, .923423), n = 5, pretty = T)
# [1] 0.0 0.2 0.4 0.6 0.8 1.0

# trim
# will trim off 10% of the tail values
# this is useful if the variable has a long tailed distribution and you want to focus on generating values near the center
x1 <- rcauchy(100)
seq_range(x1, n = 5, trim = .5)
# [1] -0.3621568  0.1304718  0.6231003  1.1157289  1.6083574

# expand
# this is the opposite or trim
# it will expand the range of predictions we want to produce or "gather"
x2 <- c(0,1)
seq_range(x2, n = 5, expand = .5)
# [1] -0.250  0.125  0.500  0.875  1.250

# let's now try to visualize our interaction model between two continous variables
# we have two continous predictors so we can imagine the model like a 3D surface
# we could display this image using geom_tile()
ggplot(grid, aes(x1, x2)) + 
        geom_tile(aes(fill = pred)) + 
        facet_wrap(~model)

# it is hard to tell any differences from these two models...
# instead of looking at the surface from the top we could look at it from either side - showing multiple slices
ggplot(grid, aes(x1, pred, color = x2, group = x2)) + 
        geom_line() + 
        facet_wrap(~model)

ggplot(grid, aes(x2, pred, color = x1, group = x1)) +
        geom_line() + 
        facet_wrap(~model)

# this shows you that interaction between two continous variables works basically the same way as for a categorical and continous variable
# an interaction says that there is not a fixed offset
# you need to consider both values of x1 and x2 simultaneously in order to predict y

# you can see that even with just two continous variables - coming up with good visualizations is hard
# that is REASONABLE:
# we should not expect it will be easy to understand how three or more simultaneuously interact with each other!

# we have saved up time by using models for exploration!
# we can gradually build up our models over time
# the model does not have to be perfect it just has to help you reveal a little more about your data


## Transformations
# you can also perform transformations inside the model formula
# we can transform certain predictors based on sqrt(), log() etc. 
# if our transformation involves *, +, ^, or - we need to wrap the formula in I()
# this will stop R from treating this transformation as part of the model specification

# if we want to see how R transforms any model we can use model_matrix to see the underlying dataset transformed into the fitted call

# data
df <- tribble(
        ~y, ~x, 
        1,1,
        2,2,
        3,3
)

# incorrectly missing I in our transformations
# result is a model with just x...not what we wanted
model_matrix(df, y ~ x^2 + x)
# # A tibble: 3 x 2
# `(Intercept)`     x
# <dbl> <dbl>
# 1          1.00  1.00
# 2          1.00  2.00
# 3          1.00  3.00

# correctly using I to model y ~ x^2 + x
model_matrix(df, y ~ I(x^2) + x)
# # A tibble: 3 x 3
# `(Intercept)` `I(x^2)`     x
# <dbl>    <dbl> <dbl>
# 1          1.00     1.00  1.00
# 2          1.00     4.00  2.00
# 3          1.00     9.00  3.00


# transformations are useful because you can use them to approximate nonlinear functions
# Taylor's theorem - you can approximate any smooth function with an infinite sum of polynomials
# this means you can use a linear function to get arbitrarily close to a smooth function by fitted an equation...
# y = a_1 + a_2 * x + a_3*x^2 + a_4 * x^3
# R provides a helper function poly() to do this for us
model_matrix(df, y ~ poly(x, 2))

# # A tibble: 3 x 3
# `(Intercept)`          `poly(x, 2)1` `poly(x, 2)2`
# <dbl>                  <dbl>         <dbl>
# 1          1.00 -0.707                         0.408
# 2          1.00 -0.0000000000000000785        -0.816
# 3          1.00  0.707                         0.408

# there is one problem using poly()
# outside the range of the data, polynomials can shoot off to positive or negative infinity
# one safer alternative is to use splines::ns()
library(splines)
model_matrix(df, y ~ ns(x, 4))
# # A tibble: 3 x 5
# `(Intercept)` `ns(x, 4)1` `ns(x, 4)2` `ns(x, 4)3` `ns(x, 4)4`
# <dbl>       <dbl>       <dbl>       <dbl>       <dbl>
# 1          1.00       0           0           0           0    
# 2          1.00       0.667       0.113       0.162      -0.108
# 3          1.00       0          -0.143       0.429       0.714


# let's see what this looks like when we try to approximate a linear function
sim5 <- tibble(
        x = seq(0, 3.5 * pi, length = 50),
        y = 4 * sin(x) + rnorm(length(x))
)

# plot the taylor theroem
# how would we model this data?
ggplot(sim5, aes(x, y)) + 
        geom_point()

# we will fit 5 models to this data with 5 higher order polynomials to see which does the best fit
mod1 <- lm(y ~ ns(x, 1), data = sim5)
mod2 <- lm(y ~ ns(x, 2), data = sim5)
mod3 <- lm(y ~ ns(x, 3), data = sim5)
mod4 <- lm(y ~ ns(x, 4), data = sim5)
mod5 <- lm(y ~ ns(x, 5), data = sim5)


# let's use grid and gather predictions to plot the predictions for each model!!
(grid <- sim5 %>% 
        data_grid(x = seq_range(x, n = 50, expand = .1)) %>% 
        gather_predictions(mod1, mod2, mod3, mod4, mod5, .pred = 'y')
)
# # A tibble: 250 x 3
# model       x     y
# <chr>   <dbl> <dbl>
# 1 mod1  -0.550  1.06 
# 2 mod1  -0.303  1.02 
# 3 mod1  -0.0561 0.975
# 4 mod1   0.191  0.932
# 5 mod1   0.438  0.889
# 6 mod1   0.684  0.845
# 7 mod1   0.931  0.802
# 8 mod1   1.18   0.759
# 9 mod1   1.42   0.715
# 10 mod1   1.67   0.672
# # ... with 240 more rows


# let's see the results!
# the more higher order polynomials we feed to the model the closer we smooth the underlying data
# however this comes at a cost - we are actually extrapolating the pattern down outside the range of our data
# if we find a new obs outside our model won;t know to keep going with the sine wave pattern
# the model can never tell if the behavior is true when going outside the range of our available data
ggplot(sim5, aes(x, y)) + 
        geom_point() + 
        geom_line(data = grid, color = 'red') + 
        facet_wrap(~model)


## Missing Values
# missing values obsviously cannot convey any information about the relationship between variables
# MODELING FUNCTIONS DROP ANY ROWS THAT CONTAIN MISSING VALUES
# R defaults behavior is to silently drop them but options(na.action = na.warn) will push a warning:
df <- tribble(
        ~x, ~y,
        1, 2.2, 
        2, NA, 
        3, 3.5, 
        4, 8.3, 
        NA, 10
)


# options to throw warnings
options(na.action = na.warn)

# test
# note that the entire rows are dropped if there is one missing value!!
mod <- lm(y ~ x, data = df)
# Warning message:
# Dropping 2 rows with missing values 

options(na.action = na.exclude)

# test - no warnings!
mod <- lm(y ~ x, data = df)

# we can always see exactly how many observations were used with nobs()
nobs(mod)
# [1] 3


## Other Model Families
# the chapter has focused exclusively on the class of linear models
# we assume a linear relatinship of the form y = B0 + B1X1 + ... BnXn
# linear models addiitonally assume resiudals have a normal distribution
# there is a large set of model classes that extend the linear model in many different ways:

# Generalized Linear Models:
# stats::glm()
# linear models assume the the repsonse is continous and the error has a normal distribution
# generalized linear models extend linear models to include non-continous repsonses
# they work by defining a distance metric based on the statistical idea of liklihood

# Generalized Additive Models
# mgcv::gam()
# these extend generalized linear models to incorporate smooth functions
# this means you can write a formula like y ~ s(x)
# which becomes an equation like y = f(x)
# and let gam() estimate what that function is (subject to some smoothness constraints to make the problem workable)

# Penalized Linear Models
# glmnet::glmnet()
# adds a penalty term to the distance that penalizes complex models (distance between parameter vector and an origin)
# the tends to make models that generalize better to new dataasets from the same population

# Robust Linear Models
# MASS::rlm()
# tweak the distance to downweight points that are very far away
# this makes them less sensitive to the presence of outliers
# this comes at the cost of being not quite as good when there are no outliers

# Trees
# rpart::rpart()
# attack the problem in complexity different than linear models
# the fit a peice-wise constant model splitting the data into smaller and smaller peices
# trees can be very powerful when a bunch are aggregated together like random forests and gradient boosting machines
# randomForest::randomForest() and xgboost::xgboost()

# these models all work similarly from a programming prospective
# once you've mastered linear models you should be able to apply the programming style to these other models
# being a skilled modeler is a mixture of some good general principles and having a big toolbox of techniques
# now that we have the general tools we can apply them to more classes and more complex problems



# chapter 19 model building -----------------------------------------------

# in the previous chapter we learned how models worked 
# we also learned some basic tools for understanding what a model is telling us about our data
# this chapter will focus on modeling with real live data

# we will take advantage of the fact you can think of a model as a partition between patterns and residuals!!
# find patterns with visualization and use these to build a fine-tuned model
# we repeat this with our fitted models residuals
# the goal is to transition from implicit knowledge in the data and your head to explicit knowledge in a quantitiative model

# very large and complex datasets can be alot of work
# there are certainly different approaches that this paritioning idea - i.e. focusing on the predictiveness of a model
# these approaches tend to produce black boxes: model does a good job at predicting but we do not know why it does

# this is a reasonable approach but it makes it hard to apply domain knowlegde to a model
# this makes it hard to see if the model will work in the long term as fundamentals change
# for real life problems we should use a combination of this approach and an automated ML type approach

# it is a challenge to know when to stop
# we need to figure out when a model is good enough and when additional work will likely pay off
# "a good artist needs to know when to stop"
# "a good artist needs to know when to start all over from scratch"

# we will use the same tools from the previous chapter but add in some live datasets
library(tidyverse); library(modelr);library(nycflights13);library(lubridate)
options(na.action = na.warn)

## why are low quality diamonds more expensive?
# in the previous chapters we've seen a surprising relationship between the quality of diamonds and thier price
# low quality diamonds (poor cuts, bad colors, bad clarity) have higher prices

# let's investigate visually
ggplot(diamonds, aes(cut, price)) + geom_boxplot()
ggplot(diamonds, aes(color, price)) + geom_boxplot()
ggplot(diamonds, aes(clarity, price)) + geom_boxplot()


## Price and Carat
# it looks like lower-quality diamond have higher prices because there is an important confoudning variable: weight
# the weight of a diamond is the single most important factor in determining the price of a diamond
ggplot(diamonds, aes(carat, price)) + 
        geom_hex(bins = 50)


# we can make it easier to see how the other attributes of a diamond affect its relative price by fitting a model to seperate out the effect of carat
# we tweak the diamonds dataset to make it more easier to work with (subsets)
# focus on diamonds smaller than 2.5 carats (99.7% of the data)
# log transform the carat and price variables
diamonds2 = diamonds %>% 
        filter(carat <= 2.5) %>% 
        mutate(lprice = log2(price), 
               lcarat = log2(carat))
# together these changes make it easier to see the relationship between carat and price
ggplot(diamonds2, aes(lcarat, lprice)) +
        geom_hex(bins = 50)

# the log transformation here is useful becuase it makes our pattern linear
# linear patterns are the easiest to understand and work with
# let's take the next step and remove the strong linear pattern
# we do this by fitting a model
mod_d = lm(lprice ~ lcarat, data = diamonds2)

# we then look at what our model is telling us about our data!
# we will overlay our predictions over the original data
grid = diamonds2 %>% 
        data_grid(carat = seq_range(carat, 20)) %>% 
        mutate(lcarat = log2(carat)) %>% 
        add_predictions(mod_d, 'lprice') %>% 
        mutate(price = 2 ^ lprice)

# visualize the results
ggplot(diamonds2, aes(carat, price)) + 
        geom_hex(bins = 50) +
        geom_line(data = grid, color = 'red', size = 1)

# this tells us something interesting about our data
# if we belive our model then the large diamonds are much cheaper than expected

# now we can look at the residuals to see if our fitted model accurately captured this pattern!
diamonds2 = diamonds2 %>% 
        add_residuals(mod_d, 'lresid')

# visualize the results
# we want our residuals to be patternless!!
ggplot(diamonds2, aes(lcarat, lresid)) + 
        geom_hex(bins = 50)

# importantly we can now redo our motivating plots using residuals instead of price!!
ggplot(diamonds2, aes(cut, lresid)) + geom_boxplot()
ggplot(diamonds2, aes(color, lresid)) + geom_boxplot()
ggplot(diamonds2, aes(clarity, lresid)) + geom_boxplot()

# now we see the relationship we expect
# as the quality of the diamond increases so does its relative price
# to interpret the y-axis, we need to think about what the residuals are telling us and thier scale
# a residual of -1 indicates that lprice was 1 unit lower than a prediction based only on weight (carat)
# so points with a value of -1 are half the expected (modeled) price
# points with residuals at 1 are twice the modeled predicted price!!


## A More Complicaed Model
# if we wanted to we could continoue to build up our model
# moving the effects we observed into the model to make them explicit
# we could include color, cut, clarity into the model so that we also make explicit the effect of these three categorical variables

# multivariate model
mod_diamonds2 = lm(
        lprice ~ lcarat + color + cut + clarity,
        data = diamonds2
)

# this model now has four predictors so it's getting harder to visualize
# each of these predictors is independent so we can plot each on seperate plots
# to write this our we will use the .model agrument in data grid
(grid = diamonds2 %>% 
        data_grid(cut, lcarat = -.515, color = 'G', clarity = 'SI1') %>% 
        add_predictions(mod_diamonds2)
)
# # A tibble: 5 x 5
# cut       lcarat color clarity  pred
# <ord>      <dbl> <chr> <chr>   <dbl>
# 1 Fair      -0.515 G     SI1      11.0
# 2 Good      -0.515 G     SI1      11.1
# 3 Very Good -0.515 G     SI1      11.2
# 4 Premium   -0.515 G     SI1      11.2
# 5 Ideal     -0.515 G     SI1      11.2

# visualize the results - cut
ggplot(grid, aes(cut, pred)) +
        geom_point()

# if the model needs variables that you haven't explicitly supplied...
# data_grid will fill them in for you with the most typical value
# continous = median, categorical = most common value
diamonds2 = diamonds2 %>% 
        add_residuals(mod_diamonds2, 'lresid2')

# plot the resiudals
ggplot(diamonds2, aes(lcarat, lresid2)) +
        geom_hex(bins = 50)

# this plot indicates that there are some diamonds with quite large residuals
# remember a residual of 2 indicates the diamond is 4x the price we expected (log transformation in model)
# we can look at the unusual values individually

# large resiudals
diamonds2 %>% 
        filter(abs(lresid2) > 2) %>% 
        add_predictions(mod_diamonds2) %>% 
        mutate(pred = round(2 ^ pred)) %>% 
        select(price, pred, carat:table, x:z) %>% 
        arrange(price)
# # A tibble: 4 x 11
# price  pred carat cut   color clarity depth table     x
# <int> <dbl> <dbl> <ord> <ord> <ord>   <dbl> <dbl> <dbl>
# 1  1186   284 0.250 Prem… G     SI2      59.0  60.0  5.33
# 2  1186   284 0.250 Prem… G     SI2      58.8  60.0  5.33
# 3  1776   412 0.290 Fair  F     SI1      55.8  60.0  4.48
# 4  2160   314 0.340 Fair  F     I1       55.8  62.0  4.72
# # ... with 2 more variables: y <dbl>, z <dbl>

#it worth spending time considering if this indicates a problem with our model or if there are errors in the data
# if there are mistakes in the data this could be an opportunity to buy diamonds that have been priced low incorrectly





## What affects the number of daily flights??
# let's work through a similiar process for another dataset
# the NYC flights data
# we can use modeling to better understand our data

# count the number of flights per day and visualize
(daily = flights %>%
        mutate(date = make_date(year, month, day)) %>% 
        group_by(date) %>% 
        summarize(n = n())
)

# # A tibble: 12 x 2
# date           n
# <date>     <int>
# 1 2013-01-01 27004
# 2 2013-02-01 24951
# 3 2013-03-01 28834
# 4 2013-04-01 28330
# 5 2013-05-01 28796
# 6 2013-06-01 28243
# 7 2013-07-01 29425
# 8 2013-08-01 29327
# 9 2013-09-01 27574
# 10 2013-10-01 28889
# 11 2013-11-01 27268
# 12 2013-12-01 28135

# number of flights per day
ggplot(daily, aes(date, n)) + 
        geom_line()


# days of week
# understanding the long term trend is challenging because there is a pattner of very strong days in week that hides other patterns
# let's look at the flight patterns by day of the week
daily = daily %>% 
        mutate(wday = wday(date, label = T))

ggplot(daily, aes(wday, n)) + 
        geom_boxplot()

# we see fewer flights on weekends because most travel is for business
# the effect is especially seen on saturday's
# one way we can remove this pattern is with modeling
# first we fit the model and display its predictions on the original data

# fit model
mod = lm(n ~ wday, data = daily)

# visualize predictions overlaid on original data
grid = daily %>% 
        data_grid(wday) %>% 
        add_predictions(mod, "n")

# visualize
ggplot(daily, aes(wday, n)) + 
        geom_boxplot() + 
        geom_point(data = grid, color = 'red', size = 4)

# next we compute and visualize the residuals
# remember models can be split into predictions and residuals for inference!

# get resiudals on original dataset
daily = daily %>% 
        add_residuals(mod)

# visualize
daily %>% 
        ggplot(aes(date, resid)) + 
        geom_ref_line(h = 0) + 
        geom_line()

# we are now seeing the deviation from "Expected" number of flights!!
# this is useful because we have removed the large day of week effect and we can see some subtler patterns

# our model starts to fail in June
# there is a strong regular pattern that our model has not captured
# drawing a plot with one line for each day of the week makes the pattern easier to see
ggplot(daily, aes(date, resid, color = wday)) + 
        geom_ref_line(h = 0) + 
        geom_line()

# our model fails to predict the number of flights on saturday
# during summer there are more flights than we expect - during the fall there are fewer

# there are some days with far fewer flights than predicted!
daily %>% filter(resid < - 100)
# A tibble: 11 x 4
# date           n wday  resid
# <date>     <int> <ord> <dbl>
# 1 2013-01-01   842 Tue    -109
# 2 2013-01-20   786 Sun    -105
# 3 2013-05-26   729 Sun    -162
# 4 2013-07-04   737 Thu    -229
# 5 2013-07-05   822 Fri    -145
# 6 2013-09-01   718 Sun    -173
# 7 2013-11-28   634 Thu    -332
# 8 2013-11-29   661 Fri    -306
# 9 2013-12-24   761 Tue    -190
# 10 2013-12-25   719 Wed    -244
# 11 2013-12-31   776 Tue    -175


# we notice that these days are mostly holidays in US!

# there seems to be some smoother long term trend over the course of a year
# we can highlight this trend with geom_smooth()
daily %>% 
        ggplot(aes(date, resid)) + 
        geom_ref_line(h = 0) +
        geom_line(color = 'grey50') + 
        geom_smooth(se = F, span = .2)

# there are fewer flights in January and more in May - September


# Seasonal Saturday Effect
# let's first tackle our failure to accurately predict the number of flights on saturday
# a good place to start is to isolate saturdays and examine thier pattern
# we use points to be sure what are true values and what is interpolation
daily %>% 
        filter(wday == 'Sat') %>% 
        ggplot(aes(date, n)) + 
        geom_point()+ 
        geom_line() + 
        scale_x_date(
                NULL, 
                date_breaks = '1 month',
                date_labels = '%b'
        )


# we suspect this pattern is caused by summer holidays:
# people don't mind travelling on Saturdays for vacation
# this lines up with summer holidays and school terms

# why are there more saturday flights in the sptring than the fall?
# possible that big holidays force travelling when the holiday is thanksgiving, xmas


# let's create a term variables that roughly captures the three school terms
# visualize
term = function(date) {
        cut(date, 
            breaks = ymd(20130101, 20130605, 20130825, 20140101),
            labels = c('spring', 'summer', 'fall')
        )
}

daily = daily %>% 
        mutate(term = term(date))

daily %>% 
        filter(wday == 'Sat') %>% 
        ggplot(aes(date, n , color = term)) +
        geom_point(alpha = 1/3) + 
        geom_line() +
        scale_x_date(
                NULL, 
                date_breaks = '1 month',
                date_labels = '%b'
        )

# its useful to see how this new variable affect the other days of the week
daily %>% 
        ggplot(aes(wday, n, color = term)) +
        geom_tufteboxplot()

# it looks like there is significant variation across the terms
# lets fit a separate day of week effect for each term
mod1 = lm(n ~ wday, data = daily)
mod2 = lm(n ~ wday * term, data = daily)

daily %>% 
        gather_residuals(without_term = mod1, with_term = mod2) %>% 
        ggplot(aes(date, resid, color = model)) + 
        geom_line(alpha = 3/4)

# we still notice some pattern in the residuals and some very extreme residuals

# let's overlay the predictions from the model onto the raw data
grid = daily %>% 
        data_grid(wday, term) %>% 
        add_predictions(mod2, 'n')

# plot predictions on actuals
ggplot(daily, aes(wday, n)) +
        geom_boxplot() +
        geom_point(data = grid, color = 'red') +
        facet_wrap(~term)

# our model is finding the mean effect but we have a lot of big outliers
# the mean tends to be far away from the typical value

# we can alleviate this problem by using a model that is robust to the effect of outliers
# we will use MASS::rlm()
# this greatly reduces the impact of the outliers on our estimates
# this model does a good job of removing the day-of-week pattern

mod3 = MASS::rlm(n ~ wday*term, data = daily)

# visualize the residuals of the model
daily %>% 
        add_residuals(mod3,'resid') %>% 
        ggplot(aes(date, resid)) + 
        geom_hline(yintercept = 0, size = 2, color = 'white') +
        geom_line()

# it is now much easier to see the long-term trend and the positive and negtive outliers outside of the trends we modeled with term and weekday



## Computed Variables
#  if you are experimenting with many models and many visualizations...
# it is a good idea to bundle the create of variables up into a function
# this avoids accidentally applying a different transformation n different places

# for example we could write:
compute_vars = function(data) {
        data %>% 
                mutate(
                        term = term(date),
                        wday = wday(date, label = T)
                )
}

# another option is to put the transformations directly in the model formula
wday2 = function(x) wday(x, label = TRUE)
mod3 = lm(n ~ wday2(date) * term(date), data = daily)

# either approach is reasonable
# making the transformed variable explicit is useful for seeing the work in your code
# including the transformations in the model keeps everything self-contained


## Time of Year: An Alternative Approach
# we used domain knowledge to whittle out the pattern of our flights data and a simple linear model to "extract" out patterns
# we could also use a more flexible model and see what overall patterns we capture
# we will use a spline model to see if we can extract the patterns across the years
library(splines)
mod_s = MASS::rlm(n ~ wday * ns(date, 5), data =  daily)

# let's visualize the results
daily %>% 
        data_grid(wday, date = seq_range(date, n = 13)) %>% 
        add_predictions(mod_s) %>% 
        ggplot(aes(date, pred, color = wday)) +
        geom_line() + 
        geom_point()

# we see a strong pattern in the number of saturday flights
# this is reassuring because we also saw that pattern in the raw data
# it is a powerful idea to model and compare the signal you get from different approaches


## Learning More about models
# we have only strached the absolute sufrace of modeling but we have gained some simple general purpose tools than can improve your own data analysis
# it is ok to start simple!
# even very simple models can make a darmatic difference in your ability to tease out interactions between variables

# these modeling chapterns are even more opionated that the rest of this book
# modeling is approached in this book a little differently that other more techinical books
# modelingg really deserves a book on its own
# read at least one of these three books:
        # Statistical Modeling: A Fresh Approach
        # Introduction to Statistical Learning
        # Applied Predictive Modeling - caret package


# chapter 20 many models with purrr and broom -----------------------------

# in this chapter we are going to learn three powerful ideas that help you work with large numbers of models

# using many simple models to better understand complex datasets
# using list-columns to store data structures in a data frame - for example a column that contains linear models
# using the broom package - turn models into tidy data

# in the example we will use a large number of simple models to parition out some of the strongest signals
# we will also investigate how model summaries can help us pick out outliers and unusual trends

# this chapter requires you to understand ideas about modeling, data structures and iteration

# we will need the following packages
library(modelr)
library(tidyverse)


## gapminder
# this example will motivate the need for many simple models
# look up Hans Rossling - a great resource!!
# the gapminder data summariezes the progression of countries over time looking at life expectancy and GDP
# we can access this data using the gapminder package
library(gapminder)
gapminder
# # A tibble: 1,704 x 6
# country     continent  year lifeExp      pop gdpPercap
# <fct>       <fct>     <int>   <dbl>    <int>     <dbl>
#         1 Afghanistan Asia       1952    28.8  8425333       779
# 2 Afghanistan Asia       1957    30.3  9240934       821
# 3 Afghanistan Asia       1962    32.0 10267083       853
# 4 Afghanistan Asia       1967    34.0 11537966       836
# 5 Afghanistan Asia       1972    36.1 13079460       740
# 6 Afghanistan Asia       1977    38.4 14880372       786
# 7 Afghanistan Asia       1982    39.9 12881816       978
# 8 Afghanistan Asia       1987    40.8 13867957       852
# 9 Afghanistan Asia       1992    41.7 16317921       649
# 10 Afghanistan Asia       1997    41.8 22227415       635
# # ... with 1,694 more rows

# in this case study we will focus on just three variables to answer the question: 'how does life expectancy change over time for each country'?
gapminder %>% 
        ggplot(aes(year, lifeExp, group = country)) +
        geom_line()

# this is a small datset that only has 1700 observations and 3 variables but it is still hard to see what is going on!
# overall it looks like life expectancy has been steadily improving
# how can we make these trends easier to see?

# one way is to fit many simple models to remove some of the larger trends to reveal more subtle trends
# we will tease these factors apart by fitting a model with a linear trend
# the model captures steady growth over time and residuals will show what is left!

# here is an example with a single country:
nz <- filter(gapminder, country == 'New Zealand')
nz %>% 
        ggplot(aes(year, lifeExp)) +
        geom_line() +
        ggtitle('Full data = ')

# fit our model
nz_mod <- lm(lifeExp ~ year, data = nz)

# add predictions back onto the original dataset
nz %>% 
        add_predictions(nz_mod) %>% 
        ggplot(aes(year, pred)) + 
        geom_line() +
        ggtitle('Linear Trend = ')


# tease out residuals by adding them back to the original data set
nz %>% 
        add_residuals(nz_mod) %>% 
        ggplot(aes(year, resid)) + 
        geom_hline(yintercept = 0, color = 'white', size = 3) + 
        geom_line() +
        ggtitle('Remaining pattern')

# how do we generalize this to every country?!?

## Nested Data
# you could imagine copying and pasting that code multiple times but there is always a better way
# never copy and paste more than 3 times!

# we could extract out the common code with a function and repeat using a map function from purrr
# in this problem we are repeating an action for each row i.e. a country
# to do this we need to learn about the nested data frame

# to create a nested dataframe we start with a grouped data frame and then nest it
(by_country <- gapminder %>% 
        group_by(country, continent) %>% 
        nest()
)
# # A tibble: 142 x 3
# country     continent data             
# <fct>       <fct>     <list>           
# 1 Afghanistan Asia      <tibble [12 × 4]>
# 2 Albania     Europe    <tibble [12 × 4]>
# 3 Algeria     Africa    <tibble [12 × 4]>
# 4 Angola      Africa    <tibble [12 × 4]>
# 5 Argentina   Americas  <tibble [12 × 4]>
# 6 Australia   Oceania   <tibble [12 × 4]>
# 7 Austria     Europe    <tibble [12 × 4]>
# 8 Bahrain     Asia      <tibble [12 × 4]>
# 9 Bangladesh  Asia      <tibble [12 × 4]>
# 10 Belgium     Europe    <tibble [12 × 4]>
# # ... with 132 more rows

# this creates a data frame that has one row per group!!
# it also has a weird column data.
# data is a list of data frames
# this seems crazy: we have a data frame with a column that is a list of other data frames!

# we can pluck out a single element from the data column you will see it contains all the data for that country!
# note data is a list we can manipulate it as a normal list!
by_country$data[[1]]
# # A tibble: 12 x 4
# year lifeExp      pop gdpPercap
# <int>   <dbl>    <int>     <dbl>
# 1  1952    28.8  8425333       779
# 2  1957    30.3  9240934       821
# 3  1962    32.0 10267083       853
# 4  1967    34.0 11537966       836
# 5  1972    36.1 13079460       740
# 6  1977    38.4 14880372       786
# 7  1982    39.9 12881816       978
# 8  1987    40.8 13867957       852
# 9  1992    41.7 16317921       649
# 10  1997    41.8 22227415       635
# 11  2002    42.1 25268405       727
# 12  2007    43.8 31889923       975

# note the difference between a standard grouped data frame and a nested data frame:
# in a grouped data frame, each row is an observation
# in a nested data frame each row is a GROUP!!
# another way to think about a nested dataset is we now have a meta-observation:
# a row that represents the complete time course for a country rather than a single point in time!


## List-Columns
# now that we have our nested data frame we can fit models over and through this list

# first create a model fitting function
country_model <- function(df) {
        lm(lifeExp ~ year, data = df)
}

# now we want to apply this to every data frame in our list column!!
# the data frames are in a list so we can use purrr::map() to apply country_model to each element!
models <- map(by_country$data, country_model)

# this splits off all these models from the original data
# instead we can add these models as another list column on our original data!!
# storing related objects in columns is a key benefit of dataframes and why list-columns are a good idea
# we are going to store all our manipulations and models together in one!

# in other words instead of creating a new object in the global enviroment we're going to create a new variable in the by_country data frame!
(by_country <- by_country %>% 
        mutate(model = map(data, country_model))
)
# # A tibble: 142 x 4
# country     continent data              model   
# <fct>       <fct>     <list>            <list>  
# 1 Afghanistan Asia      <tibble [12 × 4]> <S3: lm>
# 2 Albania     Europe    <tibble [12 × 4]> <S3: lm>
# 3 Algeria     Africa    <tibble [12 × 4]> <S3: lm>
# 4 Angola      Africa    <tibble [12 × 4]> <S3: lm>
# 5 Argentina   Americas  <tibble [12 × 4]> <S3: lm>
# 6 Australia   Oceania   <tibble [12 × 4]> <S3: lm>
# 7 Austria     Europe    <tibble [12 × 4]> <S3: lm>
# 8 Bahrain     Asia      <tibble [12 × 4]> <S3: lm>
# 9 Bangladesh  Asia      <tibble [12 × 4]> <S3: lm>
# 10 Belgium     Europe    <tibble [12 × 4]> <S3: lm>
# # ... with 132 more rows

# this has a big advantage: because all the related objects are stored together you don't need to keep them in sync when we filter!
# the semantics of the data frame will do all the work for you!
# to be clear we now have two list columns - one of the data and one of the actual model!!

# example of filtering - it will filter the list columns correctly too!
by_country %>% 
        filter(continent == 'Europe')
# # A tibble: 30 x 4
# country                continent data              model   
# <fct>                  <fct>     <list>            <list>  
# 1 Albania                Europe    <tibble [12 × 4]> <S3: lm>
# 2 Austria                Europe    <tibble [12 × 4]> <S3: lm>
# 3 Belgium                Europe    <tibble [12 × 4]> <S3: lm>
# 4 Bosnia and Herzegovina Europe    <tibble [12 × 4]> <S3: lm>
# 5 Bulgaria               Europe    <tibble [12 × 4]> <S3: lm>
# 6 Croatia                Europe    <tibble [12 × 4]> <S3: lm>
# 7 Czech Republic         Europe    <tibble [12 × 4]> <S3: lm>
# 8 Denmark                Europe    <tibble [12 × 4]> <S3: lm>
# 9 Finland                Europe    <tibble [12 × 4]> <S3: lm>
# 10 France                 Europe    <tibble [12 × 4]> <S3: lm>
# # ... with 20 more rows

# example of arranging - it will arrange all the list columns!
by_country %>% 
        arrange(continent, country)

# # A tibble: 142 x 4
# country                  continent data              model   
# <fct>                    <fct>     <list>            <list>  
# 1 Algeria                  Africa    <tibble [12 × 4]> <S3: lm>
# 2 Angola                   Africa    <tibble [12 × 4]> <S3: lm>
# 3 Benin                    Africa    <tibble [12 × 4]> <S3: lm>
# 4 Botswana                 Africa    <tibble [12 × 4]> <S3: lm>
# 5 Burkina Faso             Africa    <tibble [12 × 4]> <S3: lm>
# 6 Burundi                  Africa    <tibble [12 × 4]> <S3: lm>
# 7 Cameroon                 Africa    <tibble [12 × 4]> <S3: lm>
# 8 Central African Republic Africa    <tibble [12 × 4]> <S3: lm>
# 9 Chad                     Africa    <tibble [12 × 4]> <S3: lm>
# 10 Comoros                  Africa    <tibble [12 × 4]> <S3: lm>
# # ... with 132 more rows

# if our list of data frames and list of models were seperate objects you have to remember that whenever you reorder or subset one vector we need to do the same as the other!
# if we forget - our code will continue to work but will give us the wrong answer!

## Unnesting
# previously we computed the residuals of a single model with a single dataset
# now we have 142 data frames and 142 models
# to compute the resiudals we need to call add_residuals() with each model pair:
(by_country <- by_country %>% 
        mutate(
                resids = map2(data, model, add_residuals) # add residuals to each pair of data and model in our df
        )
)
# # A tibble: 142 x 5
# country     continent data              model    resids           
# <fct>       <fct>     <list>            <list>   <list>           
# 1 Afghanistan Asia      <tibble [12 × 4]> <S3: lm> <tibble [12 × 5]>
# 2 Albania     Europe    <tibble [12 × 4]> <S3: lm> <tibble [12 × 5]>
# 3 Algeria     Africa    <tibble [12 × 4]> <S3: lm> <tibble [12 × 5]>
# 4 Angola      Africa    <tibble [12 × 4]> <S3: lm> <tibble [12 × 5]>
# 5 Argentina   Americas  <tibble [12 × 4]> <S3: lm> <tibble [12 × 5]>
# 6 Australia   Oceania   <tibble [12 × 4]> <S3: lm> <tibble [12 × 5]>
# 7 Austria     Europe    <tibble [12 × 4]> <S3: lm> <tibble [12 × 5]>
# 8 Bahrain     Asia      <tibble [12 × 4]> <S3: lm> <tibble [12 × 5]>
# 9 Bangladesh  Asia      <tibble [12 × 4]> <S3: lm> <tibble [12 × 5]>
# 10 Belgium     Europe    <tibble [12 × 4]> <S3: lm> <tibble [12 × 5]>
# # ... with 132 more rows


# how can we plot a list of data frames?
# instead of figuring this out...
# let's turn the list of data frames back into a regular data frame
# previously we used nest() and now we will use unnest()
(resids <- unnest(by_country, resids))
# # A tibble: 1,704 x 7
# country     continent  year lifeExp      pop gdpPercap   resid
# <fct>       <fct>     <int>   <dbl>    <int>     <dbl>   <dbl>
#         1 Afghanistan Asia       1952    28.8  8425333       779 -1.11  
# 2 Afghanistan Asia       1957    30.3  9240934       821 -0.952 
# 3 Afghanistan Asia       1962    32.0 10267083       853 -0.664 
# 4 Afghanistan Asia       1967    34.0 11537966       836 -0.0172
# 5 Afghanistan Asia       1972    36.1 13079460       740  0.674 
# 6 Afghanistan Asia       1977    38.4 14880372       786  1.65  
# 7 Afghanistan Asia       1982    39.9 12881816       978  1.69  
# 8 Afghanistan Asia       1987    40.8 13867957       852  1.28  
# 9 Afghanistan Asia       1992    41.7 16317921       649  0.754 
# 10 Afghanistan Asia       1997    41.8 22227415       635 -0.534 
# # ... with 1,694 more rows

# note that each regular column is repeated once for each row in the nest column
# now that we have a regular data frame we can plot the resiudals:
resids %>% 
        ggplot(aes(year, resid)) + 
        geom_line(aes(group = country), alpha = 1/3) + 
        geom_smooth(se = F)

# faceting by continent is very insightful!
resids %>% 
        ggplot(aes(year, resid)) + 
        geom_line(aes(group = country), alpha = 1/3) + 
        geom_smooth(se = F) + 
        facet_wrap(~continent)

# it looks like we missed some mild pattern (small bump in the 80s?)
# Africa looks out of fucking wack
# the large residuals suggest we our africa specific model isn't fitting well there


## Model Quality
# instead of looking at the residuals from the model we could look at general measurements of model quality
# we will use how to do this using the broom package
# the broom package provides a general set of instructions to turn models into tidy data
# here we will use broom::glance() to extract some model quality metrics
# if we apply it to a model we get a data frame with a single row:
broom::glance(nz_mod)
# r.squared adj.r.squared     sigma statistic      p.value df    logLik      AIC    BIC
# 1 0.9535846     0.9489431 0.8043472  205.4459 5.407324e-08  2 -13.32064 32.64128 34.096
# deviance df.residual
# 1 6.469743          10

# we can use mutate and unnest to create a data frame with a row for each country
(by_country %>% 
        mutate(glance = map(model, broom::glance)) %>% 
        unnest(glance)
)
# # A tibble: 142 x 16
# country continent data  model resids r.squared adj.r.squared sigma statistic  p.value    df
# <fct>   <fct>     <lis> <lis> <list>     <dbl>         <dbl> <dbl>     <dbl>    <dbl> <int>
#         1 Afghan… Asia      <tib… <S3:… <tibb…     0.948         0.942 1.22      181   9.84e⁻ ⁸     2
# 2 Albania Europe    <tib… <S3:… <tibb…     0.911         0.902 1.98      102   1.46e⁻ ⁶     2
# 3 Algeria Africa    <tib… <S3:… <tibb…     0.985         0.984 1.32      662   1.81e⁻¹⁰     2
# 4 Angola  Africa    <tib… <S3:… <tibb…     0.888         0.877 1.41       79.1 4.59e⁻ ⁶     2
# 5 Argent… Americas  <tib… <S3:… <tibb…     0.996         0.995 0.292    2246   4.22e⁻¹³     2
# 6 Austra… Oceania   <tib… <S3:… <tibb…     0.980         0.978 0.621     481   8.67e⁻¹⁰     2
# 7 Austria Europe    <tib… <S3:… <tibb…     0.992         0.991 0.407    1261   7.44e⁻¹²     2
# 8 Bahrain Asia      <tib… <S3:… <tibb…     0.967         0.963 1.64      291   1.02e⁻ ⁸     2
# 9 Bangla… Asia      <tib… <S3:… <tibb…     0.989         0.988 0.977     930   3.37e⁻¹¹     2
# 10 Belgium Europe    <tib… <S3:… <tibb…     0.995         0.994 0.293    1822   1.20e⁻¹²     2
# # ... with 132 more rows, and 5 more variables: logLik <dbl>, AIC <dbl>, BIC <dbl>,
# #   deviance <dbl>, df.residual <int>

# this isn't quite the output we want - we still have all the list columns!
# this is the default behavior when unnest works on single row data frames
# to suppress these columns we use .drop = TRUE
(glance <- by_country %>% 
        mutate(glance = map(model, broom::glance)) %>% 
        unnest(glance, .drop = T)
)
# # A tibble: 142 x 13
# country continent r.squared adj.r.squared sigma statistic  p.value    df logLik   AIC   BIC
# <fct>   <fct>         <dbl>         <dbl> <dbl>     <dbl>    <dbl> <int>  <dbl> <dbl> <dbl>
#         1 Afghan… Asia          0.948         0.942 1.22      181   9.84e⁻ ⁸     2 -18.3  42.7  44.1 
# 2 Albania Europe        0.911         0.902 1.98      102   1.46e⁻ ⁶     2 -24.1  54.3  55.8 
# 3 Algeria Africa        0.985         0.984 1.32      662   1.81e⁻¹⁰     2 -19.3  44.6  46.0 
# 4 Angola  Africa        0.888         0.877 1.41       79.1 4.59e⁻ ⁶     2 -20.0  46.1  47.5 
# 5 Argent… Americas      0.996         0.995 0.292    2246   4.22e⁻¹³     2 - 1.17  8.35  9.80
# 6 Austra… Oceania       0.980         0.978 0.621     481   8.67e⁻¹⁰     2 -10.2  26.4  27.9 
# 7 Austria Europe        0.992         0.991 0.407    1261   7.44e⁻¹²     2 - 5.16 16.3  17.8 
# 8 Bahrain Asia          0.967         0.963 1.64      291   1.02e⁻ ⁸     2 -21.9  49.7  51.2 
# 9 Bangla… Asia          0.989         0.988 0.977     930   3.37e⁻¹¹     2 -15.7  37.3  38.8 
# 10 Belgium Europe        0.995         0.994 0.293    1822   1.20e⁻¹²     2 - 1.20  8.40  9.85
# # ... with 132 more rows, and 2 more variables: deviance <dbl>, df.residual <int>

# note there are a lot of variables not printed that are very valueable measures of model quality!
# with this data frame we can start to look for models that don't fit well!
glance %>% 
        arrange(-BIC)
# # A tibble: 142 x 13
# country  continent r.squared adj.r.squared sigma statistic p.value    df logLik   AIC   BIC
# <fct>    <fct>         <dbl>         <dbl> <dbl>     <dbl>   <dbl> <int>  <dbl> <dbl> <dbl>
# 1 Zimbabwe Africa       0.0562      -0.0381   7.21     0.596 0.458       2  -39.6  85.3  86.7
# 2 Swazila… Africa       0.0682      -0.0250   6.64     0.732 0.412       2  -38.7  83.3  84.8
# 3 Rwanda   Africa       0.0172      -0.0811   6.56     0.175 0.685       2  -38.5  83.0  84.5
# 4 Botswana Africa       0.0340      -0.0626   6.11     0.352 0.566       2  -37.7  81.3  82.8
# 5 Lesotho  Africa       0.0849      -0.00666  5.93     0.927 0.358       2  -37.3  80.6  82.1
# 6 Cambodia Asia         0.639        0.603    5.63    17.7   0.00182     2  -36.7  79.3  80.8
# 7 Namibia  Africa       0.437        0.381    4.96     7.76  0.0192      2  -35.2  76.3  77.8
# 8 South A… Africa       0.312        0.244    4.74     4.54  0.0588      2  -34.6  75.2  76.7
# 9 Zambia   Africa       0.0598      -0.0342   4.53     0.636 0.444       2  -34.1  74.1  75.6
# 10 Kenya    Africa       0.443        0.387    4.38     7.94  0.0182      2  -33.7  73.3  74.8
# # ... with 132 more rows, and 2 more variables: deviance <dbl>, df.residual <int>


# here the worst models appear to be in Africa
# let's quickly double check that with a plot
# here we have a relatively small number of observations and a discrete variable so we can use geom_jitter()
glance %>% 
        ggplot(aes(continent, BIC)) + 
        geom_jitter(width = .5)

# let's extract the countries with bad BIC and plot the data
# BIC is an esimate of TEST ERROR MSE
# the lower the error the better - the lowered the BIC the better model!!
bad_fit <- filter(glance, BIC > 75)

# plot the remaining models
gapminder %>% 
        semi_join(bad_fit, by = 'country') %>% 
        ggplot(aes(year, lifeExp, color = country)) + 
        geom_line()

# what might explain these bad models?
# HIV / AIDS and the Rwandan genocide...bummer


## List Columns
# now that we've seen a basic workflow for managing many models let's examine the details
# we will explore the list column data structure in more detail
# list columns are implicit in the definition of the data frame:
# a data frame is a named list of equal length vectors
# a list is a vector so it's always been legitmate to use a list as a column of a data frame
# however - base R doesn't make it easy to create list columns
# data.frame() treats a list as a list of columns

# data.frame example
data.frame(x = list(1:3, 3:5))

# we can prevent data.frame from doing this with I()
data.frame(
        x = I(list(1:3, 3:5)),
        y = c("1, 2","3, 4, 5")
)

# tibble alleviates this problem by being lazier
# tibble() doesn't modify its inputs
# it also gives a better print method
tibble(
        x = list(1:3, 3:5),
        y = c('1, 2', '3, 4, 5')
)

# it's even easier with tribble
# this will work out that you need a list
tribble(
        ~x, ~y,
        1:3, '1,2',
        3:5, '3,4,5'
)

# list columns are often most useful as an intermediate data structure
# they are hard to work with directly because most R functions work with atomic vectors or data frames
# the advantage of keeping related items together in a data frame is worth the hassle

# generally there are three parts of an effective list-column pipeline:
# you create the list-column using one of nest(), summarize() + list(), or mutate() + a map() function
# create other intermediate list-columns by transforming existing list columns with map(), map2(), or pmap()
# for example in the previous case study we created a list column of models transforming a list column of data frames
# you simplify the list-column back down to a data frame or atomic vector


## Creating List-Columns
# typically you won't create list-columns with tibble()
# we will create them from regular columns using three methods:
# nest() to convert a grouped data frame into a nested data frame where you have a list-column of data frames
# mutate() and vectorized functions that return a list
# summarize() and summary functions that return multiple results
# you may also create them from a named list using tibble::enframe()

# generally when creating list-columns you should make sure they are homogeneous
# each element should contain the same type of thing
# there are no checks to make sure this is true!!

## With Nesting
# nest() creates a nested data frame which is a data frame with a list column of data frames
# in a nested data frame each row is a meta observation
# the other columns give variables that define the observation(like country and continent earlier)
# the list-column of data frames gives the individual observations that make up the meta-observation

# there are two ways to nest()
# so far we've seen how to use it with a grouped data frame
# when applied to a grouped data frame nest() keeps the grouping column as is and bundles everything else into the list-column:
gapminder %>% 
        group_by(country, continent) %>% 
        nest()

# # A tibble: 142 x 3
# country     continent data             
# <fct>       <fct>     <list>           
#         1 Afghanistan Asia      <tibble [12 × 4]>
#         2 Albania     Europe    <tibble [12 × 4]>
#         3 Algeria     Africa    <tibble [12 × 4]>
#         4 Angola      Africa    <tibble [12 × 4]>
#         5 Argentina   Americas  <tibble [12 × 4]>
#         6 Australia   Oceania   <tibble [12 × 4]>
#         7 Austria     Europe    <tibble [12 × 4]>
#         8 Bahrain     Asia      <tibble [12 × 4]>
#         9 Bangladesh  Asia      <tibble [12 × 4]>
#         10 Belgium     Europe    <tibble [12 × 4]>
#         # ... with 132 more rows

# we can also use this on an ungrouped data frame
# we just need to specifiy which columns we want to nest
# these are the columns we want to convert INTO A LIST COLUMN!!!
gapminder %>% 
        nest(year:gdpPercap)

# # A tibble: 142 x 3
# country     continent data             
# <fct>       <fct>     <list>           
#         1 Afghanistan Asia      <tibble [12 × 4]>
#         2 Albania     Europe    <tibble [12 × 4]>
#         3 Algeria     Africa    <tibble [12 × 4]>
#         4 Angola      Africa    <tibble [12 × 4]>
#         5 Argentina   Americas  <tibble [12 × 4]>
#         6 Australia   Oceania   <tibble [12 × 4]>
#         7 Austria     Europe    <tibble [12 × 4]>
#         8 Bahrain     Asia      <tibble [12 × 4]>
#         9 Bangladesh  Asia      <tibble [12 × 4]>
#         10 Belgium     Europe    <tibble [12 × 4]>
#         # ... with 132 more rows


## From Vectorized Functions
# some useful functions take an atomic vector and return a list
# for example: stringr::str_split() which takes a character vector and returns a list of character vectors
# if you use that inside mutate we will get a list column
df <- tribble(
        ~x1,
        'a,b,c',
        'd,e,f,g'
)

df %>% 
        mutate(x2 = stringr::str_split(x1, ','))
# # A tibble: 2 x 2
# x1      x2       
# <chr>   <list>   
# 1 a,b,c   <chr [3]>
# 2 d,e,f,g <chr [4]>

# unnest knowns how to extract these list columns
df %>% 
        mutate(x2 = stringr::str_split(x1, ',')) %>% 
        unnest()
# # A tibble: 7 x 2
# x1      x2   
# <chr>   <chr>
#         1 a,b,c   a    
# 2 a,b,c   b    
# 3 a,b,c   c    
# 4 d,e,f,g d    
# 5 d,e,f,g e    
# 6 d,e,f,g f    
# 7 d,e,f,g g 

# if you find you are doing this alot check out tidyr::separate_rows()
# this function is a wrapper around this common pattern

# another example of this pattern is using the map() functions from purrr
sim <- tribble(
        ~f, ~params,
        'runif', list(min = -1, max = -1),
        'rnorm', list(sd = 5),
        'rpois', list(lambda = 10)
)

sim %>% 
        mutate(sims = invoke_map(f, params, n = 10))

# # A tibble: 3 x 3
# f     params     sims      
# <chr> <list>     <list>    
# 1 runif <list [2]> <dbl [10]>
# 2 rnorm <list [1]> <dbl [10]>
# 3 rpois <list [1]> <int [10]>


## From Multivalued Summaries
# one restriction of summarize() is that it only works with summary functions that return a single value
# that means you can't use it with functions like quantile() that return a vector of arbitrary length
mtcars %>% 
        group_by(cyl) %>% 
        summarize(q = quantile(mpg))
# Error in summarise_impl(.data, dots) : 
# Column `q` must be length 1 (a summary value), not 5


# we can however wrap these results in a list!
# this obeys the contract of summarize() because each summary is now a list of length 1!
mtcars %>% 
        group_by(cyl) %>% 
        summarize(q = list(quantile(mpg)))
# # A tibble: 3 x 2
# cyl q        
# <dbl> <list>   
# 1  32.0 <dbl [5]>
# 2  48.0 <dbl [5]>
# 3  64.0 <dbl [5]>

# to make useful results with unnest() we will need to also capture the probabilities
probs <- c(.01, .25, .5, .75, .99)

mtcars %>% 
        group_by(cyl) %>% 
        summarize(p = list(probs), q = list(quantile(mpg, probs))) %>% 
        unnest()
# # A tibble: 15 x 3
# cyl      p     q
# <dbl>  <dbl> <dbl>
# 1  32.0 0.0100  21.4
# 2  32.0 0.250   22.8
# 3  32.0 0.500   26.0
# 4  32.0 0.750   30.4
# 5  32.0 0.990   33.8
# 6  48.0 0.0100  17.8
# 7  48.0 0.250   18.6
# 8  48.0 0.500   19.7
# 9  48.0 0.750   21.0
# 10  48.0 0.990   21.4
# 11  64.0 0.0100  10.4
# 12  64.0 0.250   14.4
# 13  64.0 0.500   15.2
# 14  64.0 0.750   16.2
# 15  64.0 0.990   19.1



## From A Named List
# data frames with list-columns provide a solution to a common problem:
# what do you do if you want to iterate over both the contents of a list and its elements?
# instead of trying to jam everything into one object it's often easier to make a data frame:
# one column contains the elements and one column can contain the list
# we can create this data frame with enframe()
x <- list(
        a = 1:5,
        b = 3:4, 
        c = 5:6
)

(df <- enframe(x))
# # A tibble: 3 x 2
# name  value    
# <chr> <list>   
# 1 a     <int [5]>
# 2 b     <int [2]>
# 3 c     <int [2]>

# the advantage of this structure is that it generalizes in a straightforward way - 
# names are useful if you have a character vector of meta data...
# but don't help you if you have other types of data or multiple vectors

# now if we want to iterate over names and values in parallel you can use map2()
df %>% 
        mutate(
                smry = map2_chr(
                        name, 
                        value,
                        ~stringr::str_c(.x, ':', .y[1])
                )
        )
# # A tibble: 3 x 3
# name  value     smry 
# <chr> <list>    <chr>
# 1 a     <int [5]> a:1  
# 2 b     <int [2]> b:3  
# 3 c     <int [2]> c:5  

# fun example
mtcars %>% 
        group_by(cyl) %>% 
        summarize_all(funs(list))
# # A tibble: 3 x 11
# cyl mpg        disp       hp         drat       wt         qsec   vs     am     gear  carb 
# <dbl> <list>     <list>     <list>     <list>     <list>     <list> <list> <list> <lis> <lis>
# 1  32.0 <dbl [11]> <dbl [11]> <dbl [11]> <dbl [11]> <dbl [11]> <dbl … <dbl … <dbl … <dbl… <dbl…
# 2  48.0 <dbl [7]>  <dbl [7]>  <dbl [7]>  <dbl [7]>  <dbl [7]>  <dbl … <dbl … <dbl … <dbl… <dbl…
# 3  64.0 <dbl [14]> <dbl [14]> <dbl [14]> <dbl [14]> <dbl [14]> <dbl … <dbl … <dbl … <dbl… <dbl…


## Simplfying List Columns
# in order to apply most techniques we need to simplify a list column back into a normal data frame
# we can simplify our list column back into an atomic vector or set of columns
# the technique we will use to collapse our list column data frame depends on what we want:
# if we want a single value use mutate() with a map() function to create an atomic vector
# if you want many values use unnest() to convert list columns back to regular columns

## List to Vector
# if you can reduce your list column to an atomic vector then it will be a regular column
# for example - you can always summarize an object with its type and length
df = tribble(
        ~x, 
        letters[1:5],
        1:3,
        runif(5)
)

# simpliyfying a list column to one value
df %>% 
        mutate(
                type = map_chr(x, typeof), # map to a character vector typeof in a column
                length = map_int(x, length) # map to a integer vector of one equal to length in a column
        )
# # A tibble: 3 x 3
# x         type      length
# <list>    <chr>      <int>
# 1 <chr [5]> character      5
# 2 <int [3]> integer        3
# 3 <dbl [5]> double         5

# this is the same basic information we would get from the default tbl print method
# but now you can use it for filtering!
# this is a useful technique if you have a heterogenous list and want to filter out the parts that are different

# don't forget about the map_*() shortcuts
# you can use map_chr(x, 'apple') to extract the string stored in apple for each element of x
# this is usefull for pulling apart nested lists into regular columns
# use the null arguement to provide a value to use if the element is missing
df = tribble(
        ~x, 
        list(a = 1, b = 2),
        list(a = 2, c = 4)
)

# counting the number of time a string appears in each element of list column
df %>% mutate(
        a = map_dbl(x, 'a'),
        b = map_dbl(x, 'b', .null = NA_real_)
)
# A tibble: 2 x 3
# x              a     b
# <list>     <dbl> <dbl>
# 1 <list [2]>  1.00  2.00
# 2 <list [2]>  2.00 NA 


## Unnesting
# unnest() works by repeating the regular columns once for each element of the list column
# for example:
# we repated the first row four times and the second row once:
tibble(x = 1:2, y = list(1:4, 1)) %>% unnest(y)
# # A tibble: 5 x 2
# x     y
# <int> <dbl>
# 1     1  1.00
# 2     1  2.00
# 3     1  3.00
# 4     1  4.00
# 5     2  1.00


# this means that we can't simultaneously unnest two columns that contain a different number of elements!

# this is ok becuase y and z have the same number of elements in every row
(df1 = tribble(
        ~x, ~y, ~z,
        1, c('a','b'), 1:2, # two elements in y, two in z
        2, 'c', 3 # one element in y one in z
))
# # A tibble: 2 x 3
# x y         z        
# <dbl> <list>    <list>   
# 1  1.00 <chr [2]> <int [2]>
# 2  2.00 <chr [1]> <dbl [1]>

df1 %>% unnest(y, z)
# # A tibble: 3 x 3
# x y         z
# <dbl> <chr> <dbl>
# 1  1.00 a      1.00
# 2  1.00 b      2.00
# 3  2.00 c      3.00


# this doesnt work because y and z have different number of elements
(df2 = tribble(
        ~x, ~y, ~z, 
        1, 'a', 1:2, # y has one element, z has two!!
        2, c('b','c'), 3 # y has two elements, z only has one!
))
# # A tibble: 2 x 3
# x y         z        
# <dbl> <list>    <list>   
# 1  1.00 <chr [1]> <int [2]>
# 2  2.00 <chr [2]> <dbl [1]>

df2 %>% unnest(y,z)
# Error: All nested columns must have the same number of elements.


# this same principle applies when unnesting list-columns of data frames
# you can unnest multiple list-columns as long as all the data farmes in each row have the same number of rows

## Making Tidy Data with broom
# the broom package provides three general tools for turning models into tidy data frames

## broom::glance(model) reutrns a row for each model
# each column gives a model summary:
# either a measure of model quality, complexity or a combination of the two

## broom::tidy(model) returns a row for each coefficient in the model
# each column gives information about the coefficient estimate or its variability

## broom::augment(model, data) returns a row for each row in our 'data' object
# this adds extra values like residuals, and influence statistics to each row of the original data!

# broom works with a wide variety of models produced by the most popular modelling packages




# chapter 21 r markdown -----------------------------------------------------------------------

























