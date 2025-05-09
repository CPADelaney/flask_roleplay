To address the identified lint issues, here are some suggested patches in unified diff format:

### Patch 1: Replace MD5 with SHA-256

```diff
--- a/context/context_manager.py
+++ b/context/context_manager.py
@@ -102,7 +102,7 @@
 import hashlib

 def some_function():
-    hash = hashlib.md5(data).hexdigest()
+    hash = hashlib.sha256(data).hexdigest()
     return hash
```

### Patch 2: Use Secure Random Generator

```diff
--- a/logic/addiction_system_sdk.py
+++ b/logic/addiction_system_sdk.py
@@ -490,7 +490,7 @@
 import random

 def generate_random_number():
-    return random.randint(1, 100)
+    return secrets.randbelow(100) + 1
```

### Patch 3: Handle SQL Injection

```diff
--- a/data/npc_dal.py
+++ b/data/npc_dal.py
@@ -609,7 +609,7 @@
 def execute_query(query):
-    cursor.execute("SELECT * FROM users WHERE name = '%s'" % name)
+    cursor.execute("SELECT * FROM users WHERE name = ?", (name,))
```

### Patch 4: Avoid Try, Except, Pass

```diff
--- a/logic/conflict_system/conflict_tools.py
+++ b/logic/conflict_system/conflict_tools.py
@@ -904,7 +904,7 @@
 try:
     risky_operation()
-except SomeException:
-    pass
+except SomeException as e:
+    logger.error(f"An error occurred: {e}")
```

These patches address the specific issues identified by `bandit` and improve the overall security and robustness of the code.