
from langgraph.store.postgres import PostgresStore
import psycopg
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver


#---------------------------
# Checkpointer
#---------------------------
# conn=sqlite3.connect(database='chatbot.db',check_same_thread=False)
# checkpointer = SqliteSaver(conn=conn)
DB_URI="postgresql://postgres:postgres@localhost:5432/postgres_chatbot"
conn = psycopg.connect(
    DB_URI,
    autocommit=True,
    row_factory=dict_row
)
checkpointer=PostgresSaver(conn=conn)
checkpointer.setup()


DB_URI_LTM="postgresql://postgres:postgres@localhost:5442/postgres_ltm_store"
ltm_conn=psycopg.connect(
    DB_URI_LTM,
    autocommit=True
)
ltm_store=PostgresStore(conn=ltm_conn)
ltm_store.setup()


def retrive_all_threads():
    all_threads=set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]['thread_id'])

    return list(all_threads)

with conn.cursor() as cur:
    cur.execute(
        """
            create table if not exists thread_details(
            thread_id TEXT primary key,
            thread_title TEXT)
        """
    )

def add_thread_details(thread_id,title):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO thread_details (thread_id, thread_title)
            VALUES (%s, %s)
            ON CONFLICT (thread_id)
            DO UPDATE SET
            thread_title = EXCLUDED.thread_title
            """,(thread_id,title)
        )
        

def load_all_titles():
    with conn.cursor() as cur:
        cur.execute(
            """
                select thread_id,thread_title from thread_details
            """
        )
        rows = cur.fetchall()
    
    thread_details = {row["thread_id"]: row["thread_title"] for row in rows}
    return thread_details


