import pymongo
import pandas as pd
pd.options.display.width = 0
pd.set_option('display.max_rows', 500)

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["mydatabase"]
mycol = mydb["movie"]






def showall():
    cursor = mycol.find({})
    for document in cursor:
            df = pd.DataFrame.from_dict(cursor)
            print(df)

def getall():
    cursor = mycol.find({})
    return cursor

def deleterowbyid(id):
    myquery = {"_id": id}
    mycol.delete_one(myquery)
    print("success deleting")


def limitshow(num):
    myresult = mycol.find().limit(num)
    # print the result:
    for x in myresult:
        print(x)


def showallproject():
    for x in mycol.find({}, {"_id": 0, "movie": 1, "year": 1}):
        print(x)


def showbyyear(year):
    myquery = {"year": "2008" ,  "movie": { "$regex": "^The" } } # year 2018 and start with The
    for x in mycol.find(myquery, {"_id": 0, "movie": 1, "year": 1}):
        print(x)



#showallproject()
showbyyear(2008)
#showall()


