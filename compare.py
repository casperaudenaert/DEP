with open('user_recommendations_01B45481-0877-E911-80FE-001DD8B72B62_1.csv', 'r') as t1, open('user_recommendations_01B45481-0877-E911-80FE-001DD8B72B62.csv', 'r') as t2:
    fileone = t1.readlines()
    filetwo = t2.readlines()

with open('update.csv', 'w') as outFile:
    for line in filetwo:
        if line not in fileone:
            outFile.write(line)
