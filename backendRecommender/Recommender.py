from surprise import SVD
import pandas as pd
from surprise import Dataset
from surprise import Reader
from collections import defaultdict
from surprise.model_selection import cross_validate
import databaseCon

con = databaseCon.Database()
loadedData = con.selectAll2()

userGroupId = loadedData[0]
ingredientId = loadedData[1]
ratings = loadedData[2]
products = loadedData[3]

# loadedData2 = con.selectAll2()
# userGroupId = loadedData2[0]
# ingredientId = loadedData2[1]
# ratings = loadedData2[2]

def do_Predict():
    print('do_Predict ran')
    # print(userGroupId)
    # print(ingredientId)
    # print(ratings)
    ratings_dict = {'userID': userGroupId,
                    'itemID': ingredientId,
                    'rating': ratings}

    df = pd.DataFrame(ratings_dict)
    reader = Reader(rating_scale=(1, 4))
    data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
    trainset = data.build_full_trainset()
    # SVD(): single value decomposition algorithm - predict user rating for item that has yet to be rated by the user (within 1 to 5) (colaborative filtering approach)
    algo = SVD()
    algo.fit(trainset)
    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    return get_top_n(predictions)

def get_top_n(predictions, n = 20):
    # print('get_top_n ran')
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


def getRecommendedItems(uid):
    # print('getRecommendedItems ran')
    itemInfo = []
    recommendations = do_Predict()
    # print(recommendations)
    # for x in recommendations:
    #  print(x)
    for users in recommendations.items():
        # print(users[0] == uid)
        if users[0] == uid:
            for items in users[1]:
                # print(items[0])
                itemInfo.append(con.getName(items[0]))
                # itemInfo.append()
                # itemInfo.append(con.getName(items[0]))
        #         itemInfo.append(products['_id': items[0]]['productName'])
            print(itemInfo)
            # print(products)
        #     print('start printing')
        #     print(itemInfo)
        #     print(users[1])
            return(itemInfo)
            # print(users[1])
            # return(users[1])

def getAllItems():
    itemInfo = []
    for item in products:
        itemInfo.append({'_id': str(item["_id"]), 'productName': item['productName'] })
    return itemInfo;


def login(user):
    databaseCon.findUser(user)



# getRecommendedItems(1)


