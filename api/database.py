import os
from datetime import datetime

from playhouse.postgres_ext import PostgresqlExtDatabase, Model, CharField, BinaryJSONField, IntegerField, DateTimeField

peewee_database = PostgresqlExtDatabase(os.environ['FNR_DB_NAME'], **{
    'host': os.environ['FNR_DB_HOST'],
    'user': os.environ['FNR_DB_USER'],
    'password': os.environ['FNR_DB_PASSWORD'],
    'register_hstore': False
})


class BaseModel(Model):
    class Meta:
        database = peewee_database


class WordEmbedding(BaseModel):
    word = CharField(null=False)
    embedding = BinaryJSONField(null=False)
    version = IntegerField(null=False)
    inserted_at = DateTimeField(null=False, default=datetime.now)

    class Meta:
        db_table = 'word_embeddings'