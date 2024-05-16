>library(lcmm)
>library(LCTMtools)
>data<-read.csv("<PATH-TO-DATA>")
>datanew<-transform(data,timenew=scale(time))
>lcmmva<-hlme(VA~1+timenew+I(timenew^2),random=~1+timenew,ng=1,data=data.frame(datanew),subject="id")
>for (i in 2:7) {
mi <- hlme(fixed = VA ~ 1+ timenew + I(timenew^2),
mixture = ~ 1 + timenew + I(timenew^2),
random = ~ 1 + timenew,
ng = i, nwg = TRUE,
data = data.frame(datanew), subject = "id",B=lcmmva)
lin <- rbind(lin, c(i, mi$BIC))
}
>modelout <- knitr::kable(lin, col.names = c("k", "BIC"), row.names = FALSE, align = "c") 
>modelout
>model3VAnew <- hlme(fixed = VA ~1+ timenew + I(timenew^2),
mixture = ~1 + timenew + I(timenew^2),
random = ~1 + timenew,
ng = 3, nwg = T, 
idiag = FALSE, 
data = data.frame(datanew), subject = "id",B=lcmmva)
>LCTMtoolkit( model3VAnew )
>lcmm::postprob( model3VAnew)
>datnew3VA<- data.frame(timenew= seq(-1.3247, 1.0882, length = 100))
>plotpred <- predictY(model3VAnew, datnew3VA, var.time ="timenew", draws = TRUE)
>plot(plotpred, lty=1, xlab="timenew", ylab="VA", legend.loc = "topleft", cex=0.75)
>model3VAnew$pprob
