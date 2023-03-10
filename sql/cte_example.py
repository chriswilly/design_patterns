"""
"""
import sqlalchemy
from sqlalchemy.engine.url import make_url
import os


def create_connection(vendor:str,credentials:dict)->sqlalchemy.engine.Engine:
    """
    create a sqlalchemy db connection to a MS SQL-server or Oracle database
    """
    conn = None
    if vendor.lower()=='postgres':
        connection_string = 'postgresql+psycopg2://{username}:{password}@{server}:{port}/{database}'.format
    elif vendor.lower()=='microsoft':
        connection_string = 'mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}'.format
    elif vendor.lower()=='oracle':
        connection_string = 'oracle+cx_oracle://{username}:{password}@{server}:{port}/{database}'.format
    else:
        raise(NotImplementedError(f'{vendor} is not supported'))

    connection_url = make_url(connection_string(**credentials))

    engine = sqlalchemy.create_engine(connection_url)

    return engine


def sql_format(statement:str)->str:
    """
    """
    # return ' '.join([s.strip() for s in statement.split('\n')]).strip()
    return ' '.join([' '.join([word for word in line.split()]) for line in statement.splitlines()]).strip()


def sql_question(bind_variable:str,db:sqlalchemy.engine.Engine)->tuple:
    """
    """

    statement = """
    with 
    new_CTE (id, name, attr, thing)
    as (
        select col_0, col_1, col_2, col_3
        from table_thing tbl
        where
            tbl.col_1 contains ':name'
    ),

    other_CTE (id, age, rank)
    as (
        select * from other_tbl
    )
    select
        name
        ,age
        ,rank
        ,thing
    from new_CTE
    inner join other_CTE
        on new_CTE.id = other_CTE.id
    order by rank desc
    """
    with db.begin() as conn:
        result = conn.execute(sql_format(statement), bind_variable)

    return result


def main():
    """
    """
    # pw = os.getenv('PASSWORD')
    db = create_connection(
        vendor='oracle',
        credentials={
        'username':'mcw',
        'password':'solarwinds123',
        'server':'localhost',
        'port':str(1529),
        'database':'new_db'
        })

    sql_question(db=db,bind_variable='Mike')


if __name__=='__main__':
    main()
