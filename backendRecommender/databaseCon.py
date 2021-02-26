import mysql.connector
from pymongo import MongoClient
from bson.objectid import ObjectId

class Database:




    # print(collection)
    # print(client.list_database_names())

    # print(client)

    # x = db['users'].insert_many([
    #     {'username': '1', 'password': '1'},
    #     {'username': '2', 'password': '2'},
    #     {'username': '3', 'password': '3'},
    #     {'username': '4', 'password': '4'},
    #     {'username': '5', 'password': '5'},
    #     {'username': '6', 'password': '6'},
    # ])


    def __init__(self):
        self.connection = mysql.connector.connect(user='root', password='password', host='127.0.0.1', database='recommenderData', use_pure=False)
        self.cursor = self.connection.cursor()

        self.client = MongoClient(
            "mongodb+srv://admin:Pasword123@cluster0.1iwxm.mongodb.net/staticMachData?ssl=true&ssl_cert_reqs=CERT_NONE")
        self.db = self.client.mydb

        self.products = []

        for x in self.db.products.find():
            self.products.append(x);

        # test = db.tests.insert_one({'name': 'hello world'})
        # test = db.users.delete_many({})

        # users = db.users.insert_many([
        #     {'username': '1', 'password': '1'},
        #     {'username': '2', 'password': '2'},
        #     {'username': '3', 'password': '3'},
        #     {'username': '4', 'password': '4'},
        #     {'username': '5', 'password': '5'},
        #     {'username': '6', 'password': '6'},
        # ])

        # test = db.products.delete_many({})
        # products = db.products.insert_many([
        #     {'productName': 'HELP T-shirt'},
        #     {'productName': 'HELP Hoodie'},
        #     {'productName': 'HELP Jacket'},
        #     {'productName': 'MPU4 card'},
        #     {'productName': 'Student Card Replacement'},
        #     {'productName': 'HELP Lanyard'},
        #     {'productName': 'Card Sleeve'},
        #     {'productName': 'HELP Pencil'},
        #     {'productName': 'HELP Pen'},
        # ])

        # test = db.ratings.delete_many({})
        # users = ['60379bc132094475e4009cbb', '60379bc132094475e4009cbc','60379bc132094475e4009cbd','60379bc132094475e4009cbe','60379bc132094475e4009cbf','60379bc132094475e4009cc0']
        # products = ['60379d0adf24e6b057f4ef73','60379d0adf24e6b057f4ef74','60379d0adf24e6b057f4ef75','60379d0adf24e6b057f4ef76','60379d0adf24e6b057f4ef77','60379d0adf24e6b057f4ef78','60379d0adf24e6b057f4ef79','60379d0adf24e6b057f4ef7a','60379d0adf24e6b057f4ef7b']
        # seed = [[1, 4, 2, 2], [2, 4, 1, 1], [3, 1, 2, 1], [4, 3, 3, 1], [5, 5, 4, 1], [6, 1, 1, 2], [7, 4, 2, 2], [8, 4, 1, 3], [9, 2, 2, 3], [10, 3, 4, 3], [11, 2, 5, 4], [12, 2, 2, 4], [13, 2, 6, 4], [14, 4, 4, 4], [15, 2, 5, 5], [16, 4, 2, 5], [17, 2, 3, 5], [18, 4, 1, 6], [19, 3, 3, 6], [20, 5, 4, 6], [21, 2, 5, 7], [22, 2, 2, 7], [23, 3, 3, 7], [24, 5, 4, 7], [25, 1, 5, 8], [26, 2, 2, 8], [27, 5, 3, 8], [28, 1, 4, 8], [29, 4, 1, 9], [30, 1, 2, 9], [31, 3, 3, 9], [32, 5, 4, 9]]
        # ratings1 = []
        #
        # # print(users[1])
        # for i in seed:
        #     ratings1.append({'rating': i[1], 'user': users[i[2]-1], 'product': products[i[3]-1]})
        #
        # # print(ratings1);
        # test = db.ratings.delete_many({})
        # ratings = db.ratings.insert_many(ratings1)
        #
        # for x in db.ratings.find():
        #     print(x)





    def insert(self, query):
        try:
            self.cursor.execute(query)
            self.connection.commit()
        except:
            self.connection.rollback()


    def selectAll(self):
        data = []
        users = []
        items = []
        ratings = []

        self.cursor.execute("SELECT rating, Users_idUsers, Recipe_idRecipe from rating;")
        # self.cursor.execute("SELECT * from users;")
        rows = self.cursor.fetchall()
        # print(rows);
        for r in rows:
            ratings.append(r[0])
            users.append(r[1])
            items.append(r[2])

        data.append(users)
        data.append(items)
        data.append(ratings)

        return data



    def selectAll2(self):
        data2 = []
        users2 = []
        items2 = []
        ratings2 = []

        for x in self.db.ratings.find():
            users2.append(x['user']);
            items2.append(x['product']);
            ratings2.append(x['rating']);

        data2.append(users2)
        data2.append(items2)
        data2.append(ratings2)



        products2 = []
        for x in self.db.products.find():
            products2.append(x);
        data2.append(products2);


        return data2




    def getName(self, iid):
        # self.cursor.execute("SELECT recipeName FROM recipe WHERE idRecipe = " + str(iid) + ";")
        # rows = self.cursor.fetchall()
        for x in self.products:
            # print(str(x['_id']) == iid)
            if str(x['_id']) == iid:
                return x['productName']


    def findUser(self, user):
       x = self.db.users.find_one({'username': user['username'], 'password': user['password']})
       if x: x['_id'] =  str(x['_id'])
       return x


    def createUser(self, user):
        x = self.db.users.find_one({'username': user['username']})
        if not x:
            y = self.db.users.insert_one({'username': user['username'], 'password': user['password']})
            return {'_id': str(y.inserted_id), 'username': user['username'], 'password': user['password']};








