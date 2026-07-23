# Production Readiness After Module 8

Module 8 adds evaluation, request IDs, structured request logs, input limits,
provider timeouts, upload limits, metadata filtering, secret-safe configuration,
and prompt-injection boundaries. These are application foundations, not a claim
that the service is ready for an untrusted multi-tenant deployment.

## Implemented safeguards

- API keys come from environment configuration and `.env` is ignored by Git.
- Questions are limited to 2,000 characters.
- Upload formats, filenames, and a configurable 10 MiB size limit are validated.
- OpenAI requests have configurable timeouts and retry limits.
- Every response carries `X-Request-ID`.
- Request logs are JSON and contain method, path, status, duration, and request
  ID—not questions, document content, or API keys.
- Retrieval can be restricted by document ID.
- Retrieved context is marked as untrusted data in the system prompt.
- Model citations are accepted only when they identify retrieved chunks.
- The checked-in evaluation dataset measures retrieval, answer terms,
  groundedness, citations, and unsupported-question refusal separately.

## Required before a public or multi-tenant deployment

### Authentication and authorization

Add a real identity provider. Bind conversations, document records, uploads,
deletion, and retrieval filters to an authenticated user and tenant. A caller
must never gain access merely by guessing a document or conversation ID.

### Document permissions and tenant isolation

Store tenant identifiers in the document registry and vector metadata. Enforce
the tenant filter inside the service layer, not from optional client input.
Use separate Chroma collections or a production database isolation strategy
where appropriate.

### Rate limiting

Use a shared limiter such as an API gateway or Redis-backed service. An
in-memory limiter is insufficient across multiple workers and restarts.
Apply separate limits to uploads, indexing, chat requests, and provider spend.

### Monitoring

Export request latency, error rate, retrieval counts, refusal rate, token use,
provider latency, and indexing failures to the deployment's metrics and alerting
system. Avoid labels containing user text or document content.

### Storage and background processing

Move synchronous indexing to a durable job queue for large deployments. Replace
the JSON registry with a transactional database and add backups, migrations,
retention policies, malware scanning, and storage quotas.

### PostgreSQL and pgvector migration

Chroma is suitable for local development. Migrate to PostgreSQL with pgvector
when transactional document metadata, operational backups, multi-worker access,
tenant-level controls, or managed high availability become requirements. Keep
the existing vector-store service boundary so API and generation code do not
need to change.

### Security testing

Add adversarial tests for prompt injection, malicious PDFs, decompression or
parser abuse, cross-tenant identifiers, oversized multipart requests, denial of
service, leaked error details, and dependency vulnerabilities. Sandboxing or a
separate ingestion worker should be considered for untrusted documents.

## Evaluation limitations

The local evaluator uses deterministic checks: expected terms, returned source
filenames, chunk citations, and the exact refusal response. These are useful
regression signals but do not prove semantic correctness. Production evaluation
should add human-reviewed examples, retrieval relevance judgments, claim-level
groundedness review, latency/cost budgets, and representative tenant datasets.
