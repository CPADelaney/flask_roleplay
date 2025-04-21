It looks like the specific lint errors and code snippets didn’t come through. Could you please repost the lint messages along with the relevant file paths and code excerpts? For example:

```text
path/to/file.py:123:5: E231 missing whitespace after ‘,’
    foo(1,2)
    ^

--- path/to/file.py
@@ -120,7 +120,7 @@
 def foo():
-    bar(1,2)
+    bar(1, 2)
     …
```

With that info I can draft minimal unified diffs to address each issue.