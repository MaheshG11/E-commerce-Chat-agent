from llms.geminiAPI import InferGemini
from llms.Embedder import Sentence_Embedder
from Database.operations import Database

import json
import os
userInformation="User Information:{name:Mahesh, Occupation: Engineering Student,gender:Male}"
PurchaseHistory="Purchase History:Previously purchased Items include protien powder, Iphone 15 pro, Peanut butter, beardo's Face Serum, sunscreen, Hard Disk."
conversationHistory=[]
previouslyViewedProducts={"previously viewed Products":"None"}

class llmInteractions:
    
    def __init__(self) -> None:
        
        self.embedder=Sentence_Embedder()
        self.Gemini=InferGemini()
        self.database=Database(dimension= self.embedder.dimension,collectionName="productDetails",
                               metricType="IP",indexType="IVF_FLAT",port=os.getenv("MilvusPort"),
                               host=os.getenv("MilvusHost"))
        print("IP")

    async def ingest(self,data:dict):
        imgUrl= data["imgUrl"] if "imgUrl" in data else "www.example.com"
        desc=data["details"]
        price=data["price"]
        text=json.dumps(data)
        # print(type(data)) # = string
        augmentedDataList=self.Gemini.augment(text)
        
        for augmentedData in augmentedDataList:
            embedding=self.embedder.embed(text=augmentedData)
            self.database.insert(name=data["productName"],desc=desc,embedding=embedding,imgUrl=imgUrl,link="https://www.amazon.in/",price=price)

    async def inference(self,query):
        # speech to text to be integrated
        query="User's Query: "+query
        global previouslyViewedProducts
        finalResponse=self.Gemini.Inference(query=query,conversationHistory=json.dumps(conversationHistory),
                                            PurchaseHistory=PurchaseHistory,userInformation=userInformation,
                                            PreviouslyViewedProducts=json.dumps(previouslyViewedProducts))
        
        conversationHistory.append({query:finalResponse['response']})
        
        if(len(conversationHistory)>5):
            conversationHistory.remove(conversationHistory[0])
        response={"response":finalResponse['response']}
        response["products"]=[]
        if(finalResponse['rag_required']):
            search_text=finalResponse['search_phrase'] if 'search_phrase' in finalResponse else finalResponse['response']
            response["search_phrase"]=search_text
            search_vector=[self.embedder.embed(search_text)]
            print(search_text)
            topk=self.database.search(search_vector)
            products=[]
            ids_={""}
            for i in topk[0]:
                
                data=(self.database.client.get(collection_name=self.database.collectionName,ids=[i.id]))[0]
                if(data["link"] in ids_):continue
                else : ids_.add(data["link"])
                del data['embedding']
                if(data not in products):products.append(data)
            
            response["products"]=products
            previouslyViewedProducts={"Previously Viewed Products":products}
        return response
































'''
# Connect to MilvusDB
connections.connect(host="localhost", port=19530)

# Load data and preprocess
 # Replace with appropriate data loading method
# Apply preprocessing steps as needed

# Create MilvusDB collection schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, pk=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
]
schema = CollectionSchema(name="your_collection", fields=fields)

# Create MilvusDB collection
collection = Collection(schema)

# Load LLM model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Iterate over data and create embeddings
for index, row in data.iterrows():
    text = row["text"]  # Replace with appropriate column name
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

    # Insert data into MilvusDB
    collection.insert([index, text, embedding])

# Create index for efficient search
collection.create_index(field_name="embedding", index_type="IVF_FLAT")

# Example search query
query_text = "your_query_text"
query_inputs = tokenizer(query_text, return_tensors="pt")
query_outputs = model(**query_inputs)
query_embedding = query_outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

search_results = collection.search(query_embedding, top_k=5)
print(search_results)
'''