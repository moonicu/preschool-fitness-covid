suppressMessages(library(sitar))
io <- as.data.frame(iotf); wh <- as.data.frame(who0607)
write.csv(io[, c('years','sex','L.bmi','M.bmi','S.bmi')], 'iotf_bmi_lms.csv', row.names=FALSE)
write.csv(wh[, c('years','sex','L.bmi','M.bmi','S.bmi')], 'who0607_bmi_lms.csv', row.names=FALSE)
# IOTF z-scores equivalent to adult BMI 25/30 at age 18, by sex
z <- data.frame(sex=c(1,1,2,2), cut=c(25,30,25,30))
z$z <- mapply(function(s,b) LMS2z(x=18, y=b, sex=s, measure='bmi', ref='iotf'), z$sex, z$cut)
write.csv(z, 'iotf_z_cutoffs.csv', row.names=FALSE); print(z)
cat('years range iotf:', range(io$years), ' who:', range(wh$years), '\n')
