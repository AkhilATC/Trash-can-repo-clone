## Elastic search : reference for the mapping types



| **Type**                | **ES-Type** | **Description**                                                                          |
|-------------------------|-------------|------------------------------------------------------------------------------------------|
| String , VarChar        | keyword     | This is a text field that is not tokenizable: CODE001                                    |
| String , VarChar , Text | text        | This is a text field to be tokenizated: a nice text                                      |
| Integer                 | integer     | This is an Integer (32-bit): 1,2,3, or 4                                                 |
| long                    | long        | This is a long value (64-bit)                                                            |
| float                   | float       | This is a floating-point number (32-bit): 1.2, or 4.5                                    |
| double                  | double      | This is a floating point number (64 bit)                                                 |
| boolean                 | boolean     | is a Boolean value: true or false                                                        |
| date / datetime         | date        | This is a date or datetime value: 2013-12-25 , date 2013-12-25T22:21:20                  |
| bytes / binary          | binary      | This includes some bytes that are used for binary data, such as file or stream of bytes. |


###  index/mapping management in elk

```
1. CREATE INDEX
PUT <index_name>
Response:
{
  "acknowledged" : true,
  "shards_acknowledged" : true,
  "index" : "demo"
}
2. DELETE INDEX
DELETE <index-name>
Response: { "acknowledged" : true}

3. DEFINE MAPPINGS

PUT <index-name>/_mapping
{
  "properties":{
    "id":{"type":"keyword"}, // id field 
    "item":{
      "type":"object",
      "properties":{
        "name":{"type":"text"},
        "quantity" : {"type" : "integer"}
      }
    }
  }
}

4. GET MAPPING INFO
GET <index-name>/_mapping
Response:
{
  "demo" : {
    "mappings" : {
      "properties" : {
        "box" : {
          "properties" : {
            "name" : {
              "type" : "text"
            },
            "quantity" : {
              "type" : "integer"
            }
          }
        },
        "id" : {
          "type" : "keyword"
        }
      }
    }
  }
}

```



>  The most important attributes for an object are as follows:

1. properties : This is a collection of fields or objects (we can consider them
as columns in the SQL world).

2. enabled : This establishes whether or not the object should be processed. If
it's set to false, the data contained in the object is not indexed and it cannot
be searched (default true ).

3. dynamic : This allows Elasticsearch to add new field names to the object
using a reflection on the values of the inserted data. If it's set to false ,
when you try to index an object containing a new field type, it'll be rejected
silently. If it's set to strict , when a new field type is present in the object,
an error is raised, skipping the index process. The dynamic parameter
allows you to be safe about changes in the document structure (default
true ).

4. include_in_all : This adds the object values to the special _all field 
(used to aggregate the text of all document fields) (default true ).

### Mapping an alias field

The alias fields allow you to define an alias name to be resolved, as well as a query
time to simplify the call of all fields with the same meaning.
This process can be achieved executing the following actions:

1. To add this alias, we need to have a mapping that's similar to the following:
```
PUT test/_mapping
{
"properties": {
"id": {"type": "keyword"},
"date": {"type": "date"},
"customer_id": {"type": "keyword"},
"sent": {"type": "boolean"},
"item": {
"type": "object",
"properties": {
"name": {"type": "keyword"},
"quantity": {"type": "long"},
"cost": {
"type": "alias", // alias type
"path": "item.price"
},
"price": {"type": "double"},
"vat": {"type": "double"}
}
}
}
}
```
2. We can now index a record as follows:
```
PUT test/_doc/1?refresh
{
"id": "1",
"date": "2018-11-16T20:07:45Z",
"customer_id": "100",
"sent": true,
"item": [
{
"name": "tshirt",
"quantity": 10,
"price": 4.3,
"vat": 8.5
}
]
}
```
3. We can search it using the cost alias like so:
```
GET test/_search
{
"query": {
"term": {
"item.cost": 4.3
}
}
}
```

### Basic operations

#### Create index

```
PUT /myindex
{
"settings": {
"index": {
"number_of_shards": 2,
"number_of_replicas": 1
}
}
}
```
> If the index already exists, a 400 error will be returned
1. The number_of_shards , which controls the number of shards that 2^32
compose the index (every shard can store up to 2 documents)
2. The number_of_replicas , which controls the number of replications
(how many times your data is replicated in the cluster for high availability)
3. A good practice is to set this value to at least 1

> The create index command also allows for the passing of the mappings section