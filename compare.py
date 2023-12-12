from csv_diff import load_csv, compare
diff = compare(
    load_csv(open("user_recommendations_01B45481-0877-E911-80FE-001DD8B72B62_1.csv"), key="crm_Campagne_Naam_in_email"),
    load_csv(open("user_recommendations_01B45481-0877-E911-80FE-001DD8B72B62.csv"), key="crm_Campagne_Naam_in_email")
)
print(diff)