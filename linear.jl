using Random, Distributions, Turing

beta1= 4.0
beta2= -2.0
beta3= 1.0

function make_data(beta,x,sigma)
    y=x*beta
    y.+=rand(Normal(0.0,sigma),length(y))
end

beta=[beta1,beta2,beta3]

dataSigma=10.0::Float64

n_data=100

x=rand(Normal(0.0, dataSigma), 100,3)

sigmaNoise=0.25

y=make_data(beta,x,sigmaNoise)

println(size(x))
println(size(y))

priorSigma = 2.0

@model function regression(x, y, priorSigma)

    intercept ~ Normal(0, sqrt(3))
    
    sigma2 ~ truncated(Normal(0, 100), 0, Inf)
    
    beta1 ~ Normal(0, priorSigma)
    beta2 ~ Normal(0, priorSigma)
    beta3 ~ Normal(0, priorSigma)

    mu = intercept.+x*[beta1,beta2,beta3]
        
    y ~ MvNormal(mu,sqrt(sigma2))
    
end;

model = regression(x, y,priorSigma)
chain = sample(model, HMC(0.05, 10), MCMCThreads(), 1_500, 3)

describe(chain)
