# User identity propagation

OSS Spark does not automatically carry the submitting user's identity into
executor processes. Without an identity, Ranger authorization is meaningless.

This library uses a **three-tier** `ChainedUserResolver` (§17.1, §18.2):

1. `KerberosUserResolver` — short name from the current
   `UserGroupInformation`, if the Hadoop security classes are on the classpath
   AND a Kerberos principal is established.
2. `SparkPropertyUserResolver` — reads the `spark.ai.user` local property
   from `TaskContext`. This is the primary, cross-platform path.
3. `ExplicitUserResolver` — dev-only fallback from `register(..., user="...")`.

## The `execute_as` helper (§18.3)

Applications should scope a block of queries to a specific user:

```python
from spark_ai_functions import execute_as

with execute_as(spark, "alice@company.com"):
    spark.sql("SELECT ai_analyze_sentiment(body) FROM feedback").show()
```

Under the hood:

```python
spark.sparkContext.setLocalProperty("spark.ai.user", "alice@company.com")
```

Spark propagates that property to every task's `TaskContext`, which the
resolver then reads on the executor. The property is scoped to the thread that
set it, so concurrent users from different threads don't cross-contaminate.

## What NOT to do in v1 (§17.1)

- **Do not** derive user from JWT tokens, Spark Connect metadata, or session
  context. These have subtle propagation gaps — some surface only intermittently
  in long-running sessions — and are explicitly deferred to v2.
- **Do not** pass the user as a UDF parameter. It ends up in query plans and
  is trivially forgeable.

## Testing with a specific user

```python
from spark_ai_functions import register
ai = register(spark, user="tester@local")   # resolver chain still works;
                                             # the ExplicitUser resolver is a
                                             # last-resort fallback only.
```

For tests that want to exercise deny paths, stand up a custom resolver:

```python
from spark_ai_functions.governance.user_resolver import ExplicitUserResolver
from spark_ai_functions.governance.decorator import GovernanceContext, init_governance
init_governance(GovernanceContext(user_resolver=ExplicitUserResolver("bob"), ...))
```
