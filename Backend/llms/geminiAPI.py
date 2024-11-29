import google.generativeai as genai
import os
import json 




class InferGemini:
    def __init__(self) -> None:
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        genai.configure(api_key=os.getenv("GOOGLE_API_KE"))
        ke=os.getenv("GOOGLE_API_KE")
        print(f"Gemini Key: {ke}")
    def getlist(self,s):
        x=0
        y=-1
        while(s[x]!='[' and x<len(s)):
            x+=1
        x+=1
        while(s[y]!=']' and y>(-1*len(s))):y-=1
        s=s[x:y]       

        li=[]
        st=""
        for ch in s:
            if(((ord(ch)<97 or ord(ch)>122) and (ord(ch)<65 or ord(ch)>90)) and len(st)==0):
                continue
            if(ch=='\"' or ch=='\''):
                if(len(st)):
                    li.append(st)
                    st=""
            else:
                st+=ch
        return li
    
    def getJson(self,s):
        i=0
        j=len(s)-1
        while(s[i]!='{'):i+=1
        while(s[j]!='}'):j-=1
        return s[i:j+1]

    def Inference(self,query,conversationHistory="",PurchaseHistory="",userInformation="",PreviouslyViewedProducts=""):
        instructions='''
                    System: This is a conversation between user and an AI product seller. You are acting as a AI product seller. 
                            You are given conversation history, user Details and Purchase History. Do not to refer to any external 
                            knowledge base and respond as fast as possible. This system already refers to external database.

                    Output Format:
                            JSON
                            {
                              "rag_required": true|false,
                              "search_phrase":"Rag Generation keywords and phrases" 
                              "response": "The generated response"
                            }

                    Task:
                        rad_required: Determine if the query requires RAG from product knowledge base or can be answered based on the provided context.
                        search_phrase: If RAG is required, generate a search phrase to be used for searching. keep this phrase informative about the 
                        product and do not use any unnecessary words, include all necessary words and keep length of sentence short. But of length atleast 4 words
                        Response Generation: Generate a direct response based on the provided context, query if rag is required then assume 
                        that you are already showing a few products that we will get from RAG call. If additional information is required from user mention it in 
                        this response.
                    '''
        
        

        request_body=[instructions,conversationHistory,PurchaseHistory,userInformation,PreviouslyViewedProducts,query]
        response = self.model.generate_content(request_body)
        return json.loads(self.getJson(response.text))
       
    def augment(self,ProductDetailsJson:str):
        Instructions="Given Product Details, write a product description. Note: This description is to be written such that Embeddings generated using this description can be used for storing into database which then can be used for efficient and accurate vector searches for RAG implementation. Do not output anything other than product description"
        Instructions='''
        About Query: This query is to generate augmented data for a single product. The augmented data is to be used to convert it into embeddings which will perform efficiently on retrival tasks.
        System: This is an agent on a e-commerce website which augments data based on given rules for a single product, Output format is as mentioned below and product details will be given in a JSON format at the end of this query.
        Rules:
        1)Remove unnecessary words and characters: Eliminate any words or characters that do not add value to the product information.
        2)Restructure data: If necessary, reorganize the information into a more concise and informative format.
        3)Generate augmented versions: Create variations of the product name, description, and other relevant fields while preserving the core meaning and every augmented version should be appended as a new string in output list.
        4)Maintain consistency: Ensure that the augmented data remains consistent with the original product information.
        5)Remove web-addresses: Remove all the web-addresses and any information associated with them
        6)Augmented Versions Limit: Limit Augmented versions to 4 and try to maximize the difference between these version
        Output:
        1)The output shall be a list of string of augmented data and no other information.
        2) Try to keep the output as small as possible while following the rules.
        
        '''
        response = self.model.generate_content([Instructions,ProductDetailsJson])
        
        return self.getlist(response.text)      

    def query_restructure(self,query:str):
        Instructions='''
        About Query: This query is to generate augmented data for a single product. The augmented data is to be used to convert it into embeddings which will perform efficiently on retrival tasks.
        System: This is an agent on a e-commerce website which augments data based on given rules for a single product, Output format is as mentioned below and product details will be given in a JSON format at the end of this query.
        Rules:
        1)Remove unnecessary words and characters: Eliminate any words or characters that do not add value to the product information.
        2)Restructure data: If necessary, reorganize the information into a more concise and informative format.
        3)Generate augmented versions: Create variations of the product name, description, and other relevant fields while preserving the core meaning and every augmented version should be appended as a new string in output list.
        4)Maintain consistency: Ensure that the augmented data remains consistent with the original product information.
        5)Remove web-addresses: Remove all the web-addresses and any information associated with them
        6)Augmented Versions Limit: Limit Augmented versions to 4 and try to maximize the difference between these version
        Output:
        1)The output shall be a list of string of augmented data and no other information.
        2) Try to keep the output as small as possible while following the rules.
        
        '''
        response = self.model.generate_content([Instructions,query])
        return response.text       


# obj=InferGemini()
# # json.loads(obj.firstInference("what are the shoes that you sell?"))
# print(obj.firstInference("what are the shoes that you sell?"))