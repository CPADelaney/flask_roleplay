To address the `bandit` issues related to the use of weak MD5 hashes, you can modify the code to include the `usedforsecurity=False` parameter. Here's a patch for the first two instances:

```diff
--- a/context/context_manager.py
+++ b/context/context_manager.py
@@ -102,7 +102,7 @@
 
 import hashlib
 
-md5_hash = hashlib.md5()
+md5_hash = hashlib.md5(usedforsecurity=False)
```

```diff
--- a/context/memory_manager.py
+++ b/context/memory_manager.py
@@ -330,7 +330,7 @@
 
 import hashlib
 
-md5_hash = hashlib.md5()
+md5_hash = hashlib.md5(usedforsecurity=False)
```

For the SQL injection issues, consider using parameterized queries instead of string-based query construction. Here's an example patch for one instance:

```diff
--- a/data/npc_dal.py
+++ b/data/npc_dal.py
@@ -609,7 +609,7 @@
 
 def get_npc_by_name(name):
     query = "SELECT * FROM npc WHERE name = '%s'" % name
-    cursor.execute(query)
+    cursor.execute("SELECT * FROM npc WHERE name = ?", (name,))
```

These patches address specific issues and improve the security of the codebase. Consider applying similar changes to other instances of these issues.