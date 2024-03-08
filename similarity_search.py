from milvus import default_server
default_server.start()

from pymilvus import connections

connections.connect(
    host = "127.0.0.1",
    port = default_server.listen_port
)


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


from pymilvus import CollectionSchema, FieldSchema, DataType, Collection

collection_params = [
        FieldSchema(name = 'category', dtype= DataType.VARCHAR, is_primary = True, max_length=100),
        FieldSchema(name= 'item_name', dtype= DataType.VARCHAR, max_length=100),
        FieldSchema(name= 'aisle_number', dtype= DataType.VARCHAR, max_length=100),
        FieldSchema(name= 'embedding', dtype = DataType.FLOAT_VECTOR, dim = 384) 
    ]


collection_name = 'items_collection'
schema = CollectionSchema(fields=collection_params)
collections = Collection(name = collection_name, schema = schema)

index_params = {
    "index_type" : "IVF_FLAT",
    "metric_type" :"L2",
    "params": {"nlist" : 128}
}

collections.create_index(field_name = 'embedding', index_params=index_params)
collections.load()


data = {
    'Vegetables': {
        'Tomatoes': 'A12',
        'Potatoes': 'A12',
        'Spinach': 'A11'
    },
    'Frozen vegetables': {
        'Mixed vegetables': 'B10',
        'Broccoli': 'B11',
        'Carrots': 'B12'
    },
    'Ice creams': {
        'Blue bell vanilla': 'C12',
        'Hayden days': 'C12'
    }
}

for category, items in data.items():
    for item_name, aisle_number in items.items():
        embedding = model.encode(item_name)
        
        document = {
            'category': category,
            'item_name': item_name,
            'aisle_number': aisle_number,
            'embedding': embedding.tolist()  
        }
        
        collections.insert([document])



query_embedding = model.encode('Fresh Spinach')
results = collections.search(data=[query_embedding.tolist()], anns_field = "embedding", limit=1, 
                             param = {"metric_type" : "L2",
                                      "params" : {"nprobe" : 2}},
                             output_fields = ["aisle_number"]
                             )


for i, hits in enumerate(results):
  for hit in hits:
    print(hit.entity.get("aisle_number"))

