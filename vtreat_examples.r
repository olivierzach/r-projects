# vtreat example in R

library("vtreat")
packageVersion("vtreat")
 #  [1] '1.3.1'
citation('vtreat')
 #  
 #  To cite package 'vtreat' in publications use:
 #  
 #    John Mount and Nina Zumel (2018). vtreat: A Statistically Sound
 #    'data.frame' Processor/Conditioner.
 #    https://github.com/WinVector/vtreat/,
 #    https://winvector.github.io/vtreat/.
 #  
 #  A BibTeX entry for LaTeX users is
 #  
 #    @Manual{,
 #      title = {vtreat: A Statistically Sound 'data.frame' Processor/Conditioner},
 #      author = {John Mount and Nina Zumel},
 #      year = {2018},
 #      note = {https://github.com/WinVector/vtreat/, https://winvector.github.io/vtreat/},
 #    }


# categorical example
dTrainC <- data.frame(x=c('a','a','a','b','b',NA,NA),
   z=c(1,2,3,4,NA,6,NA),
   y=c(FALSE,FALSE,TRUE,FALSE,TRUE,TRUE,TRUE))
dTestC <- data.frame(x=c('a','b','c',NA),z=c(10,20,30,NA))

# help("designTreatmentsC")

treatmentsC <- designTreatmentsC(dTrainC,colnames(dTrainC),'y',TRUE,
                                 verbose=FALSE)
print(treatmentsC$scoreFrame[,c('origName', 'varName', 'code', 'rsq', 'sig', 'extraModelDegrees')])
 #    origName   varName  code         rsq        sig extraModelDegrees
 #  1        x    x_catP  catP 0.130498074 0.26400089                 2
 #  2        x    x_catB  catB 0.030345745 0.59013918                 2
 #  3        z   z_clean clean 0.237601767 0.13176020                 0
 #  4        z   z_isBAD isBAD 0.296065432 0.09248399                 0
 #  5        x  x_lev_NA   lev 0.296065432 0.09248399                 0
 #  6        x x_lev_x_a   lev 0.130005705 0.26490379                 0
 #  7        x x_lev_x_b   lev 0.006067337 0.80967242                 0

# help("prepare")

dTrainCTreated <- prepare(treatmentsC,dTrainC,pruneSig=1.0,scale=TRUE)
varsC <- setdiff(colnames(dTrainCTreated),'y')
# all input variables should be mean 0
sapply(dTrainCTreated[,varsC,drop=FALSE],mean)
 #         x_catP        x_catB       z_clean       z_isBAD      x_lev_NA 
 #   1.585994e-16  0.000000e+00  7.927952e-18 -7.926292e-18  3.965082e-18 
 #      x_lev_x_a     x_lev_x_b 
 #  -1.982154e-17  9.917546e-19
# all non NA slopes should be 1
sapply(varsC,function(c) { lm(paste('y',c,sep='~'),
   data=dTrainCTreated)$coefficients[[2]]})
 #     x_catP    x_catB   z_clean   z_isBAD  x_lev_NA x_lev_x_a x_lev_x_b 
 #          1         1         1         1         1         1         1
dTestCTreated <- prepare(treatmentsC,dTestC,pruneSig=c(),scale=TRUE)
print(dTestCTreated)
 #        x_catP     x_catB  z_clean    z_isBAD   x_lev_NA  x_lev_x_a
 #  1 -0.2380952 -0.1897682 1.194595 -0.1714286 -0.1714286 -0.2380952
 #  2  0.1785714 -0.1489924 2.951351 -0.1714286 -0.1714286  0.1785714
 #  3  0.8035714 -0.1320682 4.708108 -0.1714286 -0.1714286  0.1785714
 #  4  0.1785714  0.4336447 0.000000  0.4285714  0.4285714  0.1785714
 #      x_lev_x_b
 #  1  0.02857143
 #  2 -0.07142857
 #  3  0.02857143
 #  4  0.02857143
# numeric example
dTrainN <- data.frame(x=c('a','a','a','a','b','b',NA,NA),
   z=c(1,2,3,4,5,NA,7,NA),y=c(0,0,0,1,0,1,1,1))
dTestN <- data.frame(x=c('a','b','c',NA),z=c(10,20,30,NA))
# help("designTreatmentsN")
treatmentsN = designTreatmentsN(dTrainN,colnames(dTrainN),'y',
                                verbose=FALSE)
print(treatmentsN$scoreFrame[,c('origName', 'varName', 'code', 'rsq', 'sig', 'extraModelDegrees')])
 #    origName   varName  code          rsq       sig extraModelDegrees
 #  1        x    x_catP  catP 3.558824e-01 0.1184999                 2
 #  2        x    x_catN  catN 2.131202e-02 0.7301398                 2
 #  3        x    x_catD  catD 4.512437e-02 0.6135229                 2
 #  4        z   z_clean clean 2.880952e-01 0.1701892                 0
 #  5        z   z_isBAD isBAD 3.333333e-01 0.1339746                 0
 #  6        x  x_lev_NA   lev 3.333333e-01 0.1339746                 0
 #  7        x x_lev_x_a   lev 2.500000e-01 0.2070312                 0
 #  8        x x_lev_x_b   lev 1.110223e-16 1.0000000                 0
dTrainNTreated <- prepare(treatmentsN,dTrainN,pruneSig=1.0,scale=TRUE)
varsN <- setdiff(colnames(dTrainNTreated),'y')
# all input variables should be mean 0
sapply(dTrainNTreated[,varsN,drop=FALSE],mean) 
 #         x_catP        x_catN        x_catD       z_clean       z_isBAD 
 #   2.775558e-17  0.000000e+00 -2.775558e-17  4.857226e-17  6.938894e-18 
 #       x_lev_NA     x_lev_x_a     x_lev_x_b 
 #   6.938894e-18  0.000000e+00  7.703720e-34
# all non NA slopes should be 1
sapply(varsN,function(c) { lm(paste('y',c,sep='~'),
   data=dTrainNTreated)$coefficients[[2]]}) 
 #     x_catP    x_catN    x_catD   z_clean   z_isBAD  x_lev_NA x_lev_x_a 
 #          1         1         1         1         1         1         1 
 #  x_lev_x_b 
 #          1
dTestNTreated <- prepare(treatmentsN,dTestN,pruneSig=c(),scale=TRUE)
print(dTestNTreated)
 #    x_catP x_catN      x_catD   z_clean    z_isBAD   x_lev_NA x_lev_x_a
 #  1 -0.250  -0.25 -0.06743804 0.9952381 -0.1666667 -0.1666667     -0.25
 #  2  0.250   0.00 -0.25818161 2.5666667 -0.1666667 -0.1666667      0.25
 #  3  0.625   0.00 -0.25818161 4.1380952 -0.1666667 -0.1666667      0.25
 #  4  0.250   0.50  0.39305768 0.0000000  0.5000000  0.5000000      0.25
 #        x_lev_x_b
 #  1 -2.266233e-17
 #  2  6.798700e-17
 #  3 -2.266233e-17
 #  4 -2.266233e-17

# for large data sets you can consider designing the treatments on 
# a subset like: d[sample(1:dim(d)[[1]],1000),]