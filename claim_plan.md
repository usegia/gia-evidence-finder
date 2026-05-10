After testing the idea against those domains, I would slightly refine the proposal:

Do **not** make “Document” the central abstraction.

Make **Evidence Source** or **Source Artifact** the central abstraction. A document is one kind of source. A record field, ticket comment, resume, README, lease PDF, support thread, review, or changelog can also be a source.

The rule should be:

```text
If users search for the thing itself, model it as an entity.
If the thing explains/supports another object, model it as a source artifact.
If extracted text becomes searchable knowledge, promote it as claims.
```

So yes, documents can be first-class, but as part of a broader source/provenance layer, not as mandatory graph entities.

**Project Management**

Current entities already look like:

```text
person
team
project
ticket
comment
repository
```

Here, many “documents” are actually domain records:

```text
ticket.description
comment.body
project.summary
repository.description
```

For example:

```text
Ticket: BILL-201
Description: Enterprise customer escalated billing invoice webhook retry failures.
Comment: Customer reports webhook retries failed three times after invoice sync.
```

We should not create a separate `Document` entity for each ticket description. The ticket is already the domain entity. The text field is an evidence source.

Extraction could produce:

```text
ClaimRecord
subject = ticket:BILL-201
text = "Enterprise customer escalated billing invoice webhook retry failures."
source = ticket.description span

ClaimRecord
subject = ticket:BILL-201
text = "The ticket involves webhook retry failures after invoice sync."
source = comment.body span
```

But comments are different: comments are already first-class entities because users may query:

```text
tickets with comments from Ines about customer escalation
people commenting on mobile crash bugs
```

So in project management:

- `ticket`, `comment`, `project`, `repo` are graph entities.
- Their text fields are source artifacts.
- PRDs, incident docs, changelogs, and READMEs are attached source artifacts.
- Claims enrich ticket/project/repo/person search.

This domain shows why “every document is an entity” is too heavy.

**People Search**

Entities:

```text
person
organization
```

Edges:

```text
worked_at
founded
invested_in
advised
followed
```

Sources:

```text
resume
LinkedIn profile
personal bio
company page
press article
GitHub profile
conference speaker page
blog post
```

Example source:

```text
Resume attached to person:Jane
"Built distributed systems and AI deployment infrastructure before starting VectorForge."
```

Extracted claim:

```text
subject = person:Jane
text = "Jane built AI deployment infrastructure."
source = resume span
```

Another source:

```text
Press article
"Jane Carter founded VectorForge after working on infrastructure at Stripe."
```

This is trickier. It may mention multiple entities and relationships. The safe v1 behavior should be:

```text
ClaimProposal attached to person:Jane
"Jane founded VectorForge."

ClaimProposal attached to person:Jane
"Jane previously worked on infrastructure at Stripe."
```

Later, after we trust extraction more, this could create reviewed `EdgeProposal`s:

```text
person:Jane --founded--> org:VectorForge
person:Jane --worked_at--> org:Stripe
```

But v1 should not auto-create edges. It should produce claims first.

This domain also proves another important rule: person-level claims are sometimes weaker than organization or edge evidence. For example, TraverseDB already has profile logic where “from AI companies” should prefer employer organization evidence over person claims. Auto-extracted claims must respect that.

**Apartment Search**

Entities:

```text
listing
unit
building
neighborhood
landlord
amenity
transit_station
```

Sources:

```text
listing description
lease PDF
building rules
inspection report
agent email
tenant reviews
amenity guide
neighborhood guide
```

Example:

```text
Listing description:
"Pet-friendly apartment with in-unit laundry. No broker fee."

Lease PDF:
"Cats are allowed with written approval. Dogs are not permitted."

Review:
"Street noise is noticeable on weekends."
```

Claims:

```text
subject = listing:apt_123
text = "The apartment allows cats with approval."
source = lease PDF span
confidence = high

subject = listing:apt_123
text = "Dogs are not permitted."
source = lease PDF span
negated = true

subject = listing:apt_123
text = "The apartment has in-unit laundry."
source = listing description span

subject = listing:apt_123
text = "Street noise may be noticeable on weekends."
source = tenant review span
confidence = lower
```

But structured fields should remain structured:

```text
price = 3200
bedrooms = 2
available_at = 2026-06-01
neighborhood = Williamsburg
```

We should not extract price or bedrooms as claims when they are reliable fields. Claims are best for policies, amenities, qualitative facts, caveats, and evidence-backed assertions.

Apartment search also shows why source reliability matters:

```text
lease PDF > official listing > agent email > tenant review
```

So claims need provenance and reliability metadata, not just text.

**Revised Recommendation**

The best model is:

```text
SourceArtifact
  kind: document | record_field | comment_thread | external_page | pdf | email | review
  source_uri
  content_hash
  source_system
  reliability
  observed_at

SourceSpan
  source_artifact_id
  offsets
  text
  heading/path/context

SourceAttachment
  source_artifact_id
  attached_to: ObjectRef
  role: describes | mentions | supports | field_source

ClaimProposal
  subject: ObjectRef
  text
  source_spans
  confidence
  polarity
  status: proposed | auto_promoted | rejected | stale

ClaimRecord
  promoted durable searchable claim
```

The public DX can still say “documents” for simplicity:

```python
db.sources.ingest(
    kind="readme",
    uri="README.md",
    text=readme,
    attach_to=ObjectRef.entity("repository", "ultrachess"),
    extract_claims=True,
)
```

But internally it should be source artifacts, not only documents.

For existing records:

```python
db.records.upsert(
    entity=ticket,
    extract_claims_from=["description"],
)
```

For comments:

```python
db.records.upsert_many(
    entities=comments,
    extract_claims_from=["body"],
    attach_claims_to="ticket_via:comment_on",
)
```

That last part is important: sometimes the source is a comment entity, but the useful claim belongs to the ticket.

**Positioning**

The positioning should not be:

> “TraverseDB has document search.”

It should be:

> “TraverseDB turns source text into source-backed semantic graph evidence.”

That is much stronger.

It means customers can bring messy text, but TraverseDB converts it into inspectable, benchmarkable search artifacts:

```text
raw source text -> verified claim proposals -> searchable claims -> explainable results
```

**Main Pitfalls**

The biggest design risks are:

- Treating all documents as graph entities.
- Extracting claims from fields where structured data should be used instead.
- Attaching a claim to the wrong subject.
- Letting weak claims overpower stronger graph/edge evidence.
- Creating too many trivial claims.
- Ignoring source reliability and recency.
- Failing to mark stale claims when sources change.
- Auto-promoting relationship facts that should be edge proposals.
- Treating absence in a document as a true negative.

**My Concrete View**

The earlier proposal is directionally right, but I would rename and generalize it:

Not:

```text
First-class Documents -> attach to entities -> extract claims
```

Better:

```text
First-class Evidence Sources -> attach to graph scopes -> extract ClaimProposals -> promote to ClaimRecords
```

Documents are just one source type.

This gives us the cleanest product and DX:

- Project management works naturally with existing tickets/comments.
- People search works with resumes, bios, and press articles.
- Apartment search works with listings, lease PDFs, reviews, and policy docs.
- TraverseDB stays a semantic graph database, not a document database.
- Claims become the safe intermediate layer before we ever suggest edges, concepts, or schema changes.

I think this is the best option: native, simple enough, and much more general than “documents as entities.”