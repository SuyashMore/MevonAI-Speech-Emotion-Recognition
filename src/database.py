import pymongo  
import time

CLIENT_ADDR= "#"
DATABASE = "call_center"
COLLECTION = "speech_transaction"

def appendData(operator,data):
    client = pymongo.MongoClient(CLIENT_ADDR)
    print("Client Received")
    # Access database 
    mydatabase = client[DATABASE] 
    print("DB Received")
    # Access collection of the database 
    mycollection=mydatabase[COLLECTION] 
    print("Collection Received") 

    rec = {"operator":operator,
            "payload":data,
            "time":time.time()} 
    
    # inserting the data in the database 
    print("Inserting to DB:")
    rec = mycollection.insert_one(rec) 
    print("DB Access Complete.")
# print(rec)

def getData(op):
    client = pymongo.MongoClient(CLIENT_ADDR)
    # Access database 
    mydatabase = client[DATABASE] 
    # Access collection of the database 
    mycollection=mydatabase[COLLECTION] 

    for i in mydatabase.myTable.find({"operator": op}):
        print(i) 

    print("Itr Success....")



if __name__=="__main__":
    # appendData("Hello WOrld")
    getData("Ajay1667")
    # passs