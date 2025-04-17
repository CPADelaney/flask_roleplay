# init_db_script.py
import asyncio
from db.schema_and_seed import initialize_all_data

if __name__ == "__main__":
    asyncio.run(initialize_all_data())
    print("Database created and default data seeded.")
