
# Linear regression
Linear regression is a statistical method used to model the relationship between a dependent variable (often denoted as "y") and one or more independent variables (often denoted as "x"). The basic idea of linear regression is to find the straight line that best fits the data points in a scatter plot.

The most common form of linear regression is simple linear regression, which models the relationship between two variables:


where y is the dependent variable, x is the independent variable, m is the slope, and b is the intercept.

Given a set of input data (), the goal of linear regression is to find the values of m and b that best fit the data

The values of m and b are chosen to minimize the "sum of squared errors" (SSE) 
.

Taking the partial derivatives with respect to m and b, set them equal to 0, and solve for m and b, we get:

m = sum((x - x_mean) * (y - y_mean)) / sum((x - x_mean)^2)
b = y_mean - m * x_mean

Multiple linear regression is a more general form of linear regression that models the relationship between multiple independent variables and one dependent variable. The formula for the best-fit hyperplane in multiple linear regression is:

y = w₀ + w₁·x₁ + w₂·x₂ + ... + wₙ·xₙ = Xᵀ·W



## How to run:

Run fast api:
> uvicorn main:app --reload

Test Simple Linear Regression:
> curl -X POST "http://127.0.0.1:8000/predict/?medinc=3.5"

Start Docker
> docker build -t linear-model .
> docker run -p 8000:80 linear-model

Start docker compose
> docker-compose up --build
> docker compose --env-file .env up


