using Random, Distributions, Turing

beta1= 0.4
beta2= -0.2
beta3= 1.0

function make_data(beta,x,sigma)
    y=x*beta
    y.+=rand(Normal(0.0,sigma),length(y))
end

beta=[beta1,beta2,beta3]

dataSigma=10.0::Float64

nData=100

x=rand(Normal(0.0, dataSigma), nData,3)

sigmaNoise=0.25


y=make_data(beta,x,sigmaNoise)


@model function regression(x, y)

    intercept ~ Normal(0, sqrt(3))
    
    sigma2 ~ truncated(Normal(0, 100), 0, Inf)

    priorSigma ~ Exponential(1)
    beta ~ MvNormal([0.0,0.0,0.0], priorSigma)

    mu = intercept.+x*beta

    
    
    y ~ MvNormal(mu,sqrt(sigma2))
    
end

model = regression(x, y)
chain = sample(model, NUTS(), MCMCThreads(), 1_500, 3)

describe(chain)
