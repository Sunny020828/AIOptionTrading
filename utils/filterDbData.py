import mysql.connector
import os

def clean_sensitive_data():
    config = {
        "host": os.getenv("MYSQLHOST", "127.0.0.1"),
        "port": int(os.getenv("MYSQLPORT", 3306)),
        "user": os.getenv("MYSQLUSER", "root"),
        "password": os.getenv("MYSQLPASSWORD", "Strong!pass"),
        "database": "history",  # 请替换为你的数据库名
        "autocommit": False,
    }

    # 建议排查的敏感词列表
    sensitive_keywords = [
        "反送中", "雨傘革命", "佔中", "光復香港", 
        "五大訴求", "港獨", "台獨", "疆獨", "習近平","民意","民運","民主"
    ]
    
    # 需要检查的表名和列名
    target_table = "news" 
    target_column = "title"

    try:
        connection = mysql.connector.connect(**config)
        cursor = connection.cursor()
        
        total_deleted = 0
        stats = {}

        print(f"--- 开始清理数据库: {target_table} ---")

        for word in sensitive_keywords:
            # 使用 LIKE 进行模糊匹配查询
            query = f"DELETE FROM {target_table} WHERE {target_column} LIKE %s"
            pattern = f"%{word}%"
            
            cursor.execute(query, (pattern,))
            affected_rows = cursor.rowcount
            
            stats[word] = affected_rows
            total_deleted += affected_rows
            print(f"关键词 '{word}': 删除了 {affected_rows} 行")

        # 提交更改
        connection.commit()
        
        print("\n--- 清理完成统计 ---")
        print(f"总计删除行数: {total_deleted}")
        for word, count in stats.items():
            if count > 0:
                print(f" - {word}: {count} 条")

    except Exception as e:
        print(f"发生错误: {e}")
        if 'connection' in locals():
            connection.rollback()
    finally:
        if 'connection' in locals():
            cursor.close()
            connection.close()

if __name__ == "__main__":
    clean_sensitive_data()