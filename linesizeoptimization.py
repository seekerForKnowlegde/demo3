# import pandas as pd
# import numpy as np
# from scipy.optimize import minimize

# # Step 1: Input Data (you can replace with a CSV or DB read)
# data = {
#     'Policy_ID': ['P1', 'P2', 'P3', 'P4','P5'],
#     'Gross_Premium': [1316, 740, 9432, 2305,1765],
#     'Loss': [592, 414,7074 ,2005,1676],
#     'Original_LineSize': [0.45, 0.5, 0.2, 0.4,0.35],
#     'Renewal': [1, 1, 1, 1,1],  # Initially all policies are renewed
# }
# df = pd.DataFrame(data)

# # Step 2: Define Constraints
# target_gross_premium = 150000
# min_gross_premium = 140000
# max_increase_pct = 0.2
# max_decrease_pct = 0.2
# max_non_renewals = 0.3  # Max 30% non-renewed
# max_pml_pct = 0.8       # Max 30% loss to premium

# # Step 3: Objective Function (Minimize Loss Ratio)
# def loss_ratio(vars, premiums, losses):
#     line_sizes = vars[:len(premiums)]
#     renewals = vars[len(premiums):]
#     adjusted_premiums = line_sizes * premiums * renewals
#     adjusted_losses = line_sizes * losses * renewals
#     total_premium = np.sum(adjusted_premiums)
#     total_loss = np.sum(adjusted_losses)
#     if total_premium == 0:
#         return np.inf
#     return total_loss / total_premium

# # Step 4: Initial Guess
# n = len(df)
# initial_line_sizes = df['Original_LineSize'].values
# initial_renewals = df['Renewal'].values
# initial_guess = np.concatenate([initial_line_sizes, initial_renewals])

# # Step 5: Bounds
# bounds = []
# for original in initial_line_sizes:
#     bounds.append((
#         original * (1 - max_decrease_pct),
#         original * (1 + max_increase_pct)
#     ))
# bounds.extend([(0, 1)] * n)  # For renewals

# # Step 6: Constraints
# def premium_constraint(vars):
#     line_sizes = vars[:n]
#     renewals = vars[n:]
#     total_premium = np.sum(line_sizes * df['Gross_Premium'] * renewals)
#     return total_premium - min_gross_premium

# def non_renewal_constraint(vars):
#     renewals = vars[n:]
#     return max_non_renewals * n - (n - np.sum(renewals))

# def pml_constraint(vars):
#     line_sizes = vars[:n]
#     renewals = vars[n:]
#     adjusted_premiums = line_sizes * df['Gross_Premium'] * renewals
#     adjusted_losses = line_sizes * df['Loss'] * renewals
#     total_premium = np.sum(adjusted_premiums)
#     total_loss = np.sum(adjusted_losses)
#     return max_pml_pct * total_premium - total_loss

# constraints = [
#     {'type': 'ineq', 'fun': premium_constraint},
#     {'type': 'ineq', 'fun': non_renewal_constraint},
#     {'type': 'ineq', 'fun': pml_constraint},
# ]

# # Step 7: Run Optimization
# result = minimize(
#     loss_ratio,
#     initial_guess,
#     args=(df['Gross_Premium'].values, df['Loss'].values),
#     method='SLSQP',
#     bounds=bounds,
#     constraints=constraints
# )

# # Step 8: Apply and Show Result
# df['Optimized_LineSize'] = result.x[:n]
# df['Optimized_Renewal'] = np.round(result.x[n:])

# # Step 9: Output
# print("\n🔧 Optimization Status:", result.message)
# print("\n📋 Final Optimized Portfolio:\n")
# print(df[['Policy_ID', 'Gross_Premium', 'Loss', 'Original_LineSize', 'Optimized_LineSize', 'Optimized_Renewal']])






# import pandas as pd
# import numpy as np
# from scipy.optimize import minimize

# def optimize_portfolio(df, target_gross_premium, max_pml_pct, max_line_change_pct, max_non_renewal_pct):
#     n = len(df)
    
#     # Initial values
#     initial_line_sizes = df['Original_LineSize'].values
#     initial_renewals = df['Renewal'].values
#     initial_guess = np.concatenate([initial_line_sizes, initial_renewals])

#     # Objective: Minimize loss ratio
#     def loss_ratio(vars, premiums, losses):
#         line_sizes = vars[:n]
#         renewals = vars[n:]
#         adj_premiums = line_sizes * premiums * renewals
#         adj_losses = line_sizes * losses * renewals
#         total_premium = np.sum(adj_premiums)
#         total_loss = np.sum(adj_losses)
#         if total_premium == 0:
#             return np.inf
#         return total_loss / total_premium

#     # Bounds
#     bounds = []
#     for original in initial_line_sizes:
#         min_ls = original * (1 - max_line_change_pct)
#         max_ls = original * (1 + max_line_change_pct)
#         bounds.append((min_ls, max_ls))
#     bounds.extend([(0, 1)] * n)  # renewals

#     # Constraints
#     def premium_constraint(vars):
#         line_sizes = vars[:n]
#         renewals = vars[n:]
#         total_premium = np.sum(line_sizes * df['Gross_Premium'].values * renewals)
#         return total_premium - target_gross_premium

#     def pml_constraint(vars):
#         line_sizes = vars[:n]
#         renewals = vars[n:]
#         adj_losses = np.sum(line_sizes * df['Loss'].values * renewals)
#         adj_premiums = np.sum(line_sizes * df['Gross_Premium'].values * renewals)
#         return max_pml_pct * adj_premiums - adj_losses

#     def non_renewal_constraint(vars):
#         renewals = vars[n:]
#         return max_non_renewal_pct * n - (n - np.sum(renewals))

#     constraints = [
#         {'type': 'ineq', 'fun': premium_constraint},
#         {'type': 'ineq', 'fun': pml_constraint},
#         {'type': 'ineq', 'fun': non_renewal_constraint},
#     ]

#     result = minimize(
#         loss_ratio,
#         initial_guess,
#         args=(df['Gross_Premium'].values, df['Loss'].values),
#         method='SLSQP',
#         bounds=bounds,
#         constraints=constraints
#     )

#     df['Optimized_LineSize'] = result.x[:n]
#     df['Optimized_Renewal'] = np.round(result.x[n:])
#     return df[['Policy_ID', 'Gross_Premium', 'Loss', 'Original_LineSize', 'Optimized_LineSize', 'Optimized_Renewal']]


# if __name__ == "__main__":
    # Sample data (you can replace with CSV load)
    # df_sample = pd.DataFrame({
    #     'Policy_ID': ['P1', 'P2', 'P3', 'P4', 'P5'],
    #     'Gross_Premium': [1316, 740, 9432, 2305,1765],
    #     'Loss': [592, 414,7074 ,2005,1676],
    #     'Original_LineSize': [0.45, 0.5, 0.2, 0.4,0.35],
    #     'Renewal': [1, 1, 1, 1, 1]
    # })

    # # User-defined constraints
    # target_gross_premium = 150000
    # max_pml_pct = 0.4  # 80%
    # max_line_change_pct = 0.4  # ±20%
    # max_non_renewal_pct = 0.5  # 30%

    # optimized_df = optimize_portfolio(
    #     df_sample,
    #     target_gross_premium,
    #     max_pml_pct,
    #     max_line_change_pct,
    #     max_non_renewal_pct
    # )
    # optimized_loss_ratio=ca

    # print("\n📊 Optimized Portfolio:\n")
    # print(optimized_df)


import numpy as np
import pandas as pd
from scipy.optimize import minimize

def optimize_portfolio(df, target_gross_premium, max_pml_pct, max_line_change_pct, max_non_renewal_pct):
    n = len(df)

    # Initial guess: use original values
    x0 = np.concatenate([df['Original_LineSize'].values, df['Renewal'].values])

    # Bounds for each variable
    bounds = []
    for i in range(n):
        orig_line = df.loc[i, 'Original_LineSize']
        bounds.append((max(orig_line - max_line_change_pct, 0), min(orig_line + max_line_change_pct, 1)))
    for i in range(n):
        bounds.append((0, 1))  # Renewal can range from 0 to 1

    def objective(x):
        lines = x[:n]
        renews = x[n:]
        total_loss = np.sum(lines * renews * df['Loss'])
        total_premium = np.sum(lines * renews * df['Gross_Premium'])
        if total_premium == 0:
            return np.inf
        return total_loss / total_premium  # loss ratio

    def constraint_total_premium(x):
        lines = x[:n]
        renews = x[n:]
        return np.sum(lines * renews * df['Gross_Premium']) - target_gross_premium

    def constraint_max_pml(x):
        lines = x[:n]
        renews = x[n:]
        total_loss = np.sum(lines * renews * df['Loss'])
        total_premium = np.sum(lines * renews * df['Gross_Premium'])
        if total_premium == 0:
            return -1  # force constraint fail
        return max_pml_pct - (total_loss / total_premium)

    def constraint_max_non_renewal(x):
        renews = x[n:]
        non_renewed = np.sum(1 - renews)
        return max_non_renewal_pct * n - non_renewed

    constraints = [
        {'type': 'ineq', 'fun': constraint_total_premium},
        {'type': 'ineq', 'fun': constraint_max_pml},
        {'type': 'ineq', 'fun': constraint_max_non_renewal},
    ]

    result = minimize(objective, x0, bounds=bounds, constraints=constraints)

    df['Optimized_LineSize'] = result.x[:n]
    df['Optimized_Renewal'] = np.round(result.x[n:])

    return df

def calculate_loss_ratio(df, line_col, renew_col):
    total_loss = np.sum(df[line_col] * df[renew_col] * df['Loss'])
    total_premium = np.sum(df[line_col] * df[renew_col] * df['Gross_Premium'])
    return (total_loss / total_premium)  if  total_premium != 0 else 0
 

if __name__ == "__main__":
    df_sample = pd.DataFrame({
        'Policy_ID': ['P1', 'P2', 'P3', 'P4', 'P5'],
        'Gross_Premium': [1316, 740, 9432, 2305, 1765],
        'Loss': [592, 414, 7074, 2005, 1676],
        'Original_LineSize': [0.45, 0.5, 0.2, 0.4, 0.35],
        'Renewal': [1, 1, 1, 1, 1],
    })

    # User constraints
    target_gross_premium = 15000
    max_pml_pct = 0.4
    max_line_change_pct = 0.2
    max_non_renewal_pct = 0.5

    # Calculate original loss ratio
    original_loss_ratio = calculate_loss_ratio(df_sample, 'Original_LineSize', 'Renewal')

    # Optimize
    optimized_df = optimize_portfolio(df_sample.copy(), target_gross_premium, max_pml_pct, max_line_change_pct, max_non_renewal_pct)

    # Calculate optimized loss ratio
    optimized_loss_ratio = calculate_loss_ratio(optimized_df, 'Optimized_LineSize', 'Optimized_Renewal')

    # Print results
    print("\nOptimized Portfolio:")
    print(optimized_df[['Policy_ID', 'Gross_Premium', 'Loss', 'Original_LineSize', 'Optimized_LineSize', "Renewal",'Optimized_Renewal']])
    print(f"\n🔹 Original Loss Ratio:   {original_loss_ratio:.2f}%")
    print(f"🔹 Optimized Loss Ratio: {optimized_loss_ratio:.2f}%")