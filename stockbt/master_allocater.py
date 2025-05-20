import math
from scipy.stats import norm

def gaussian_prob(x, mu, sigma):
    return norm.cdf(x, mu, sigma)

def bellcurve_allocation(dict):
    balance = dict["balance"]
    stocks = dict["stocks"]
    sum_calc_list = []
    return_dict = {}
    sigma = 0.1
    for i in range(len(stocks)):
        print(i)
        stock = stocks[i]
        ticker = stock["ticker"]
        risk = stock["risk"]
        print(risk)
        pn_risk = 1 - (risk / 100)
        print(pn_risk)
        
        print(prob_up)
        alloc = math.floor(balance * (prob_up**2))
        return_dict[ticker] = alloc
        sum_calc_list.append(alloc)
        print(sum_calc_list)

    bomboclat = balance - sum(sum_calc_list)
    print(f"Remainder (bomboclat): {bomboclat}")
    return_dict["bomboclat"] = bomboclat
    
    return return_dict

    """ now risk in this case is basically whats the percentage chance that this stock will go down, over a given time period. One thing we can try to do it use a modify Gaussian distribution to calculate this. 
    A normal gaussian is a bell curve, that can be modeled as a the probablity graph if each step is a 50/50 chance to go up or down. Mathematically, this is just simply a type of pascals triangle:
                                        1/2
                                        1/4 1/4
                                        1/8 1/4 1/8
                                        ....
                                        
                                        1/2^n
                                        1/(2^(n+1)) 1/(2^(n+1))
                                        1/(2^(n+2)) (1/(2^(n+1)) + 1/(2^(n+1)))*1/2 1/(2^(n+2))
                                        
                                        now for a n/100 risk to go down, the triangle should be smth like:
                                        
                                        n/100 (100 - n)/100
                                        (n/100)^2 2*n/100*(100 - n)/100 (100 - n)/100^2
                                        .....
                                        so on and so forth.
                                        Now after converting this to a gaussian curve, we can use this to calculate the percentage chance of the stock going up, and then we can use that to calculate the allocation for each stock. 
                                        
                                        Lets say 1 stock has an 80% chance to go up, and the other has a 40% chance to go up. Convert these to 0.8 and 0.4, now sqaure the two to get 0.64 and 0.16. Thats how much of the balance should be allocated to each stock. 
                                        
    
    
    
    
    """

def input_to_dict():
    balance = float(input("Enter the balance: "))
    num_stocks = int(input("Enter the number of stocks: "))
    stocks = []
    for i in range(num_stocks):
        ticker = input(f"Enter the ticker for stock {i+1}: ")
        risk = float(input(f"Enter the risk for stock {i+1}: "))
        stocks.append({"ticker": ticker, "risk": risk})
    return {"balance": balance, "stocks": stocks}



if __name__ == "__main__":
    dict = input_to_dict()
    return_dict = bellcurve_allocation(dict)
    print(return_dict)

