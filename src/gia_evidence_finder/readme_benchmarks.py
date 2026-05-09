from __future__ import annotations

from dataclasses import dataclass

from gia_evidence_finder.contracts import (
    BenchmarkCase,
    BenchmarkCuration,
    BenchmarkSuite,
    DocumentSpan,
    EvidenceDocument,
    EvidenceFacet,
    EvidenceRelation,
    IntentSpec,
    SpanKind,
)
from gia_evidence_finder.parsing import MarkdownSpanParser


@dataclass(frozen=True)
class ReadmeExcerpt:
    id: str
    repository: str
    source_url: str
    license_name: str
    text: str


README_EXCERPTS: tuple[ReadmeExcerpt, ...] = (
    ReadmeExcerpt(
        id="react",
        repository="facebook/react",
        source_url="https://raw.githubusercontent.com/facebook/react/main/README.md",
        license_name="MIT",
        text="""# React

React is a JavaScript library for building user interfaces.

* Declarative: React makes it painless to create interactive UIs. Design simple
  views for each state in your application, and React will efficiently update
  and render just the right components when your data changes.
* Component-Based: Build encapsulated components that manage their own state,
  then compose them to make complex UIs.
* Learn Once, Write Anywhere: React does not make assumptions about the rest of
  your technology stack, so you can develop new features without rewriting
  existing code.

## Installation

React has been designed for gradual adoption from the start, and you can use as
little or as much React as you need.

* Use Quick Start to get a taste of React.
* Add React to an Existing Project to use as little or as much React as you need.
* Create a New React App if you are looking for a powerful JavaScript toolchain.
""",
    ),
    ReadmeExcerpt(
        id="kubernetes",
        repository="kubernetes/kubernetes",
        source_url="https://raw.githubusercontent.com/kubernetes/kubernetes/master/README.md",
        license_name="Apache-2.0",
        text="""# Kubernetes

Kubernetes, also known as K8s, is an open source system for managing
containerized applications across multiple hosts. It provides basic mechanisms
for deployment, maintenance, and scaling of applications.

Kubernetes builds upon a decade and a half of experience at Google running
production workloads at scale using a system called Borg, combined with
best-of-breed ideas and practices from the community.

Kubernetes is hosted by the Cloud Native Computing Foundation. If your company
wants to help shape the evolution of container-packaged, dynamically scheduled,
and microservices-oriented technologies, consider joining the CNCF.

## To start using K8s

See our documentation on kubernetes.io.
""",
    ),
    ReadmeExcerpt(
        id="django",
        repository="django/django",
        source_url="https://raw.githubusercontent.com/django/django/main/README.rst",
        license_name="BSD-3-Clause",
        text="""# Django

Django is a high-level Python web framework that encourages rapid development
and clean, pragmatic design.

All documentation is in the docs directory and online at
https://docs.djangoproject.com/en/stable/. If you are just getting started,
first read docs/intro/install.txt for instructions on installing Django.

Next, work through the tutorials in order, starting with
docs/intro/tutorial01.txt and docs/intro/tutorial02.txt.

If you want to set up an actual deployment server, read
docs/howto/deployment/index.txt for instructions.

Docs are updated rigorously. If you find problems in the docs, fill out a
ticket at https://code.djangoproject.com/newticket.
""",
    ),
    ReadmeExcerpt(
        id="fastapi",
        repository="fastapi/fastapi",
        source_url="https://raw.githubusercontent.com/fastapi/fastapi/master/README.md",
        license_name="MIT",
        text="""# FastAPI

FastAPI framework, high performance, easy to learn, fast to code, ready for
production.

## Features

* Fast: Very high performance, on par with NodeJS and Go, thanks to Starlette
  and Pydantic.
* Fast to code: Increase the speed to develop features by about 200% to 300%.
* Fewer bugs: Reduce about 40% of human developer induced errors.
* Intuitive: Great editor support, completion everywhere, and less time
  debugging.
* Robust: Get production-ready code with automatic interactive documentation.
* Standards-based: Based on and fully compatible with the open standards for
  APIs, OpenAPI and JSON Schema.
""",
    ),
    ReadmeExcerpt(
        id="pytorch",
        repository="pytorch/pytorch",
        source_url="https://raw.githubusercontent.com/pytorch/pytorch/main/README.md",
        license_name="BSD-style",
        text="""# PyTorch

PyTorch is a Python package that provides two high-level features:

* Tensor computation, like NumPy, with strong GPU acceleration.
* Deep neural networks built on a tape-based autograd system.

You can reuse your favorite Python packages such as NumPy, SciPy, and Cython to
extend PyTorch when needed.

Our trunk health and continuous integration signals can be found at
hud.pytorch.org.
""",
    ),
    ReadmeExcerpt(
        id="rust",
        repository="rust-lang/rust",
        source_url="https://raw.githubusercontent.com/rust-lang/rust/master/README.md",
        license_name="MIT OR Apache-2.0",
        text="""# Rust

This is the main source code repository for Rust. It contains the compiler,
standard library, and documentation.

## Why Rust

* Performance: Fast and memory-efficient, suitable for critical services,
  embedded devices, and easy integration with other languages.
* Reliability: The rich type system and ownership model ensure memory and
  thread safety, reducing bugs at compile-time.
* Productivity: Comprehensive documentation, a compiler committed to great
  diagnostics, and advanced tooling including Cargo, rustfmt, Clippy, and
  rust-analyzer.

## Quick Start

Read Installation from The Book.
""",
    ),
    ReadmeExcerpt(
        id="vscode",
        repository="microsoft/vscode",
        source_url="https://raw.githubusercontent.com/microsoft/vscode/main/README.md",
        license_name="MIT",
        text="""# Visual Studio Code - Open Source

This repository, Code - OSS, is where Microsoft develops the Visual Studio Code
product together with the community.

Visual Studio Code is a distribution of the Code - OSS repository with
Microsoft-specific customizations released under a traditional Microsoft
product license.

Visual Studio Code combines the simplicity of a code editor with what developers
need for their core edit-build-debug cycle.

## Contributing

There are many ways to participate in this project:

* Submit bugs and feature requests, and help verify them as they are checked in.
* Review source code changes.
* Review the documentation and make pull requests for anything from typos to
  new content.
""",
    ),
    ReadmeExcerpt(
        id="ansible",
        repository="ansible/ansible",
        source_url="https://raw.githubusercontent.com/ansible/ansible/devel/README.md",
        license_name="GPL-3.0",
        text="""# Ansible

Ansible is a radically simple IT automation system. It handles configuration
management, application deployment, cloud provisioning, ad-hoc task execution,
network automation, and multi-node orchestration.

Ansible makes complex changes like zero-downtime rolling updates with load
balancers easy.

## Design Principles

* Have an extremely simple setup process with a minimal learning curve.
* Manage machines quickly and in parallel.
* Avoid custom agents and additional open ports; be agentless by leveraging the
  existing SSH daemon.
* Describe infrastructure in a language that is both machine and human friendly.
* Focus on security and easy auditability, review, and rewriting of content.
* Manage new remote machines instantly, without bootstrapping any software.
* Allow module development in any dynamic language, not just Python.
* Be usable as non-root.
""",
    ),
)


def popular_readme_benchmark_suite() -> BenchmarkSuite:
    documents = _readme_documents()
    cases: list[BenchmarkCase] = []
    cases.extend(_react_cases(documents["react"]))
    cases.extend(_kubernetes_cases(documents["kubernetes"]))
    cases.extend(_django_cases(documents["django"]))
    cases.extend(_fastapi_cases(documents["fastapi"]))
    cases.extend(_pytorch_cases(documents["pytorch"]))
    cases.extend(_rust_cases(documents["rust"]))
    cases.extend(_vscode_cases(documents["vscode"]))
    cases.extend(_ansible_cases(documents["ansible"]))
    return BenchmarkSuite(
        id="popular_readme_v1",
        name="Popular README intent-to-span benchmark",
        description=(
            "Curated intent evidence cases from README excerpts of popular open-source "
            "repositories. Excerpts are small, source-attributed, and include hard negatives."
        ),
        cases=tuple(cases),
        metadata={
            "document_count": str(len(documents)),
            "case_count": str(len(cases)),
        },
    )


def hard_readme_benchmark_suite() -> BenchmarkSuite:
    documents = _readme_documents()
    cases: list[BenchmarkCase] = []
    cases.extend(_react_hard_cases(documents["react"]))
    cases.extend(_kubernetes_hard_cases(documents["kubernetes"]))
    cases.extend(_django_hard_cases(documents["django"]))
    cases.extend(_fastapi_hard_cases(documents["fastapi"]))
    cases.extend(_pytorch_hard_cases(documents["pytorch"]))
    cases.extend(_rust_hard_cases(documents["rust"]))
    cases.extend(_vscode_hard_cases(documents["vscode"]))
    cases.extend(_ansible_hard_cases(documents["ansible"]))
    return BenchmarkSuite(
        id="hard_readme_v1",
        name="Hard README intent-to-span benchmark",
        description=(
            "Adversarial README evidence cases emphasizing paraphrases, nearby technical "
            "anchors, and unsupported intents. Built from the same source-attributed "
            "popular README excerpts as popular_readme_v1."
        ),
        cases=tuple(cases),
        metadata={
            "document_count": str(len(documents)),
            "case_count": str(len(cases)),
            "source_suite": "popular_readme_v1",
        },
    )


def adversarial_readme_benchmark_suite() -> BenchmarkSuite:
    documents = _readme_documents()
    cases: list[BenchmarkCase] = []
    cases.extend(_react_adversarial_cases(documents["react"]))
    cases.extend(_kubernetes_adversarial_cases(documents["kubernetes"]))
    cases.extend(_django_adversarial_cases(documents["django"]))
    cases.extend(_fastapi_adversarial_cases(documents["fastapi"]))
    cases.extend(_pytorch_adversarial_cases(documents["pytorch"]))
    cases.extend(_rust_adversarial_cases(documents["rust"]))
    cases.extend(_vscode_adversarial_cases(documents["vscode"]))
    cases.extend(_ansible_adversarial_cases(documents["ansible"]))
    return BenchmarkSuite(
        id="adversarial_readme_v1",
        name="Adversarial README claim benchmark",
        description=(
            "Hard unsupported, polarity, and near-miss cases designed to expose false "
            "support and wrong top-span behavior. This suite is diagnostic and may be "
            "ahead of the current default ranker."
        ),
        cases=tuple(cases),
        metadata={
            "document_count": str(len(documents)),
            "case_count": str(len(cases)),
            "source_suite": "popular_readme_v1",
            "diagnostic": "true",
        },
    )


def relation_readme_benchmark_suite() -> BenchmarkSuite:
    documents = _readme_documents()
    cases = _relation_sensitive_cases(documents)
    return BenchmarkSuite(
        id="relation_readme_v1",
        name="Relation-sensitive README claim benchmark",
        description=(
            "Diagnostic README cases that stress subject, predicate, and modifier binding. "
            "The suite pairs supported claims with unsupported relation reversals or "
            "attribute transfers from nearby spans."
        ),
        cases=cases,
        metadata={
            "document_count": str(len(documents)),
            "case_count": str(len(cases)),
            "source_suite": "popular_readme_v1",
            "diagnostic": "true",
            "focus": "relation_binding",
        },
    )


def readme_benchmark_suites() -> tuple[BenchmarkSuite, ...]:
    return (
        popular_readme_benchmark_suite(),
        hard_readme_benchmark_suite(),
        adversarial_readme_benchmark_suite(),
        relation_readme_benchmark_suite(),
    )


def readme_benchmark_suite_by_id(suite_id: str) -> BenchmarkSuite:
    suites = {suite.id: suite for suite in readme_benchmark_suites()}
    try:
        return suites[suite_id]
    except KeyError as exc:
        raise ValueError(f"unknown README benchmark suite {suite_id!r}") from exc


def _readme_documents() -> dict[str, EvidenceDocument]:
    return {
        excerpt.id: MarkdownSpanParser().parse(
            excerpt.text,
            document_id=excerpt.id,
            source=excerpt.source_url,
        )
        for excerpt in README_EXCERPTS
    }


def _relation_sensitive_cases(
    documents: dict[str, EvidenceDocument],
) -> tuple[BenchmarkCase, ...]:
    react = documents["react"]
    kubernetes = documents["kubernetes"]
    django = documents["django"]
    fastapi = documents["fastapi"]
    pytorch = documents["pytorch"]
    rust = documents["rust"]
    vscode = documents["vscode"]
    ansible = documents["ansible"]

    react_existing = _span(react, "Add React to an Existing Project")
    react_new_app = _span(react, "Create a New React App", kind=SpanKind.BULLET)
    kubernetes_borg = _span(kubernetes, "system called Borg")
    kubernetes_cncf = _span(kubernetes, "hosted by the Cloud Native Computing Foundation")
    django_install = _span(django, "docs/intro/install.txt")
    django_deployment = _span(django, "deployment server")
    django_ticket = _span(django, "fill out a")
    fastapi_performance = _span(fastapi, "Very high performance", kind=SpanKind.BULLET)
    fastapi_standards = _span(fastapi, "OpenAPI and JSON Schema")
    pytorch_packages = _span(pytorch, "NumPy, SciPy, and Cython")
    pytorch_ci = _span(pytorch, "hud.pytorch.org")
    rust_reliability = _span(rust, "ownership model ensure memory")
    rust_productivity = _span(rust, "Cargo, rustfmt, Clippy")
    vscode_oss = _span(vscode, "develops the Visual Studio Code product together")
    vscode_distribution = _span(vscode, "Microsoft-specific customizations")
    ansible_agentless = _span(ansible, "agentless by leveraging")
    ansible_human = _span(ansible, "machine and human friendly")

    return (
        _case(
            "relation.react_existing_project_option",
            react,
            "React option for an existing project",
            "Find evidence that React can be added to an existing project.",
            ("Add React to an Existing Project",),
            (react_existing,),
            forbidden=(react_new_app,),
            facets=(EvidenceFacet("existing project", ("Existing Project",)),),
            relations=(
                EvidenceRelation(
                    name="react_existing_project_option",
                    subject_phrases=("React",),
                    predicate_phrases=("Add", "use"),
                    object_phrases=("Existing Project",),
                ),
            ),
        ),
        _case(
            "relation.react_new_app_for_existing_project",
            react,
            "new app option for existing project",
            (
                "Find evidence that Create a New React App is the option for adding "
                "React to an existing project."
            ),
            ("Create a New React App existing project",),
            (),
            forbidden=(react_existing, react_new_app),
            facets=(
                EvidenceFacet("new app", ("Create a New React App",)),
                EvidenceFacet("existing", ("existing project",)),
            ),
            relations=(
                EvidenceRelation(
                    name="new_app_existing_project_option",
                    subject_phrases=("Create a New React App",),
                    predicate_phrases=("option", "adding"),
                    object_phrases=("existing project",),
                ),
            ),
            expect_abstain=True,
            min_support_score=0.55,
        ),
        _case(
            "relation.kubernetes_borg_lineage",
            kubernetes,
            "Kubernetes builds on Google Borg experience",
            "Find evidence that Kubernetes builds on Google's Borg production experience.",
            ("Kubernetes builds upon Google production workloads Borg",),
            (kubernetes_borg,),
            forbidden=(kubernetes_cncf,),
            facets=(
                EvidenceFacet("google", ("Google",)),
                EvidenceFacet("borg", ("Borg",)),
            ),
            relations=(
                EvidenceRelation(
                    name="kubernetes_borg_lineage",
                    subject_phrases=("Kubernetes",),
                    predicate_phrases=("builds upon", "builds on"),
                    object_phrases=("Google", "Borg", "production workloads"),
                ),
            ),
        ),
        _case(
            "relation.kubernetes_cncf_ran_borg",
            kubernetes,
            "CNCF ran Borg workloads",
            "Find evidence that CNCF ran Google's Borg production workloads.",
            ("CNCF Google Borg production workloads",),
            (),
            forbidden=(kubernetes_borg, kubernetes_cncf),
            facets=(
                EvidenceFacet("cncf", ("CNCF", "Cloud Native Computing Foundation")),
                EvidenceFacet("borg", ("Borg",)),
            ),
            relations=(
                EvidenceRelation(
                    name="cncf_borg_operation",
                    subject_phrases=("CNCF", "Cloud Native Computing Foundation"),
                    predicate_phrases=("ran", "running"),
                    object_phrases=("Borg", "production workloads"),
                ),
            ),
            expect_abstain=True,
            min_support_score=0.55,
        ),
        _case(
            "relation.django_docs_ticket",
            django,
            "Django docs problems reported through ticket",
            "Find evidence that problems in Django docs should be reported through a ticket.",
            ("problems in docs fill out ticket",),
            (django_ticket,),
            forbidden=(django_install,),
            facets=(
                EvidenceFacet("docs problems", ("problems in the docs",)),
                EvidenceFacet("ticket", ("ticket",)),
            ),
            relations=(
                EvidenceRelation(
                    name="docs_problems_ticket",
                    subject_phrases=("problems in the docs",),
                    predicate_phrases=("fill out", "reported"),
                    object_phrases=("ticket",),
                ),
            ),
        ),
        _case(
            "relation.django_deployment_first_read",
            django,
            "deployment docs are first getting-started read",
            "Find evidence that deployment docs are the first document for new Django users.",
            ("getting started first read deployment instructions",),
            (),
            forbidden=(django_install, django_deployment),
            facets=(
                EvidenceFacet("first", ("first read",)),
                EvidenceFacet("deployment", ("deployment",)),
            ),
            relations=(
                EvidenceRelation(
                    name="deployment_first_read",
                    subject_phrases=("deployment docs", "deployment"),
                    predicate_phrases=("first read", "getting started"),
                    object_phrases=("new users", "Django"),
                ),
            ),
            expect_abstain=True,
            min_support_score=0.55,
        ),
        _case(
            "relation.fastapi_openapi_json_schema",
            fastapi,
            "FastAPI compatible with OpenAPI and JSON Schema",
            "Find evidence that FastAPI is compatible with OpenAPI and JSON Schema.",
            ("compatible with OpenAPI and JSON Schema",),
            (fastapi_standards,),
            forbidden=(fastapi_performance,),
            facets=(EvidenceFacet("standards", ("OpenAPI", "JSON Schema")),),
            relations=(
                EvidenceRelation(
                    name="fastapi_standards",
                    subject_phrases=("FastAPI",),
                    predicate_phrases=("compatible", "based on"),
                    object_phrases=("OpenAPI", "JSON Schema"),
                ),
            ),
        ),
        _case(
            "relation.fastapi_pydantic_matches_node_go",
            fastapi,
            "Pydantic itself matches NodeJS and Go performance",
            "Find evidence that Pydantic itself is on par with NodeJS and Go for performance.",
            ("Pydantic on par with NodeJS and Go performance",),
            (),
            forbidden=(fastapi_performance,),
            insufficient_context=(fastapi_performance,),
            facets=(
                EvidenceFacet("pydantic", ("Pydantic",)),
                EvidenceFacet("node go", ("NodeJS", "Go")),
            ),
            relations=(
                EvidenceRelation(
                    name="pydantic_performance",
                    subject_phrases=("Pydantic",),
                    predicate_phrases=("performance", "on par"),
                    object_phrases=("NodeJS", "Go"),
                    forbidden_bridge_phrases=("thanks to", "based on", "derived from"),
                ),
            ),
            expect_abstain=True,
            min_support_score=0.55,
        ),
        _case(
            "relation.pytorch_python_packages_extend",
            pytorch,
            "Python packages extend PyTorch",
            "Find evidence that NumPy, SciPy, and Cython can extend PyTorch.",
            ("NumPy SciPy Cython extend PyTorch",),
            (pytorch_packages,),
            forbidden=(pytorch_ci,),
            facets=(EvidenceFacet("packages", ("NumPy", "SciPy", "Cython")),),
            relations=(
                EvidenceRelation(
                    name="packages_extend_pytorch",
                    subject_phrases=("NumPy", "SciPy", "Cython"),
                    predicate_phrases=("extend", "reuse"),
                    object_phrases=("PyTorch",),
                ),
            ),
        ),
        _case(
            "relation.pytorch_packages_publish_ci",
            pytorch,
            "Python packages publish trunk health signals",
            "Find evidence that NumPy, SciPy, and Cython publish PyTorch trunk health signals.",
            ("NumPy SciPy Cython trunk health continuous integration",),
            (),
            forbidden=(pytorch_packages, pytorch_ci),
            facets=(
                EvidenceFacet("packages", ("NumPy", "SciPy", "Cython")),
                EvidenceFacet("ci", ("trunk health", "continuous integration")),
            ),
            relations=(
                EvidenceRelation(
                    name="packages_publish_ci",
                    subject_phrases=("NumPy", "SciPy", "Cython"),
                    predicate_phrases=("publish", "signals"),
                    object_phrases=("trunk health", "continuous integration"),
                ),
            ),
            expect_abstain=True,
            min_support_score=0.55,
        ),
        _case(
            "relation.rust_ownership_safety",
            rust,
            "Rust ownership model ensures memory and thread safety",
            "Find evidence that Rust's ownership model ensures memory and thread safety.",
            ("ownership model memory thread safety compile-time",),
            (rust_reliability,),
            forbidden=(rust_productivity,),
            facets=(
                EvidenceFacet("ownership", ("ownership model",)),
                EvidenceFacet("safety", ("memory", "thread safety")),
            ),
            relations=(
                EvidenceRelation(
                    name="ownership_safety",
                    subject_phrases=("ownership model",),
                    predicate_phrases=("ensure", "reducing"),
                    object_phrases=("memory", "thread safety"),
                ),
            ),
        ),
        _case(
            "relation.rust_tools_ensure_memory_safety",
            rust,
            "Rust tools ensure memory and thread safety",
            (
                "Find evidence that Cargo, rustfmt, Clippy, and rust-analyzer ensure "
                "memory and thread safety."
            ),
            ("Cargo rustfmt Clippy rust-analyzer memory thread safety",),
            (),
            forbidden=(rust_reliability, rust_productivity),
            facets=(
                EvidenceFacet("tools", ("Cargo", "Clippy", "rust-analyzer")),
                EvidenceFacet("safety", ("memory", "thread safety")),
            ),
            relations=(
                EvidenceRelation(
                    name="tools_safety",
                    subject_phrases=("Cargo", "Clippy", "rust-analyzer"),
                    predicate_phrases=("ensure", "reducing"),
                    object_phrases=("memory", "thread safety"),
                ),
            ),
            expect_abstain=True,
            min_support_score=0.55,
        ),
        _case(
            "relation.vscode_code_oss_community_development",
            vscode,
            "Code OSS community development repository",
            "Find evidence that Code - OSS is where Microsoft develops VS Code with the community.",
            ("Code OSS Microsoft develops Visual Studio Code product with community",),
            (vscode_oss,),
            forbidden=(vscode_distribution,),
            facets=(EvidenceFacet("community", ("together with the community",)),),
            relations=(
                EvidenceRelation(
                    name="code_oss_development",
                    subject_phrases=("Code - OSS",),
                    predicate_phrases=("develops",),
                    object_phrases=("Visual Studio Code product", "community"),
                ),
            ),
        ),
        _case(
            "relation.vscode_code_oss_has_distribution_customizations",
            vscode,
            "Code OSS has distribution customizations",
            (
                "Find evidence that Code - OSS itself has Microsoft-specific product "
                "license customizations."
            ),
            ("Code OSS itself Microsoft-specific customizations product license",),
            (),
            forbidden=(vscode_oss, vscode_distribution),
            insufficient_context=(vscode_oss, vscode_distribution),
            facets=(EvidenceFacet("customizations", ("Microsoft-specific customizations",)),),
            relations=(
                EvidenceRelation(
                    name="code_oss_customizations",
                    subject_phrases=("Code - OSS",),
                    predicate_phrases=("has", "includes"),
                    object_phrases=("Microsoft-specific customizations", "product license"),
                    forbidden_bridge_phrases=("distribution of", "released under"),
                ),
            ),
            expect_abstain=True,
            min_support_score=0.55,
        ),
        _case(
            "relation.ansible_agentless_ssh",
            ansible,
            "Ansible is agentless by using SSH",
            "Find evidence that Ansible avoids custom agents by leveraging SSH.",
            ("agentless leveraging existing SSH daemon avoid custom agents",),
            (ansible_agentless,),
            forbidden=(ansible_human,),
            facets=(
                EvidenceFacet("agentless", ("agentless", "avoid custom agents")),
                EvidenceFacet("ssh", ("SSH daemon",)),
            ),
            relations=(
                EvidenceRelation(
                    name="ansible_agentless_ssh",
                    subject_phrases=("Ansible",),
                    predicate_phrases=("agentless", "avoid custom agents"),
                    object_phrases=("SSH daemon",),
                ),
            ),
        ),
        _case(
            "relation.ansible_ssh_is_human_language",
            ansible,
            "SSH daemon is the human-friendly infrastructure language",
            (
                "Find evidence that the existing SSH daemon is the human-friendly "
                "infrastructure language."
            ),
            ("SSH daemon machine human friendly infrastructure language",),
            (),
            forbidden=(ansible_agentless, ansible_human),
            facets=(
                EvidenceFacet("ssh", ("SSH daemon",)),
                EvidenceFacet("human", ("human friendly",)),
            ),
            relations=(
                EvidenceRelation(
                    name="ssh_human_language",
                    subject_phrases=("SSH daemon",),
                    predicate_phrases=("is", "describes"),
                    object_phrases=("human friendly", "infrastructure language"),
                ),
            ),
            expect_abstain=True,
            min_support_score=0.55,
        ),
    )


def _react_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    component = _span(document, "Build encapsulated components")
    gradual = _span(document, "gradual adoption")
    existing = _span(document, "Add React to an Existing Project")
    app = _span(document, "Create a New React App", kind=SpanKind.BULLET)
    return (
        _case(
            "react.components",
            document,
            "component composition with local state",
            "Find evidence that React components manage state and compose into complex UIs.",
            ("encapsulated components manage their own state and compose",),
            (component,),
            facets=(
                EvidenceFacet("components", ("components",)),
                EvidenceFacet("state", ("state",)),
            ),
        ),
        _case(
            "react.gradual_adoption",
            document,
            "use only as much React as needed",
            "Find evidence that React can be adopted gradually in an existing project.",
            ("use as little or as much React as needed",),
            (gradual, existing),
            forbidden=(app,),
            facets=(EvidenceFacet("gradual", ("gradual adoption", "little or as much")),),
        ),
    )


def _react_hard_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    stack = _span(document, "does not make assumptions about the rest")
    existing = _span(document, "Add React to an Existing Project")
    new_app = _span(document, "Create a New React App", kind=SpanKind.BULLET)
    return (
        _case(
            "hard.react.stack_agnostic",
            document,
            "framework can fit an existing technology stack",
            "Find evidence that React does not force assumptions about the surrounding stack.",
            ("technology stack assumptions develop features without rewriting",),
            (stack,),
            forbidden=(new_app,),
            facets=(
                EvidenceFacet("stack", ("technology stack",)),
                EvidenceFacet("no rewrite", ("without rewriting existing code",)),
            ),
        ),
        _case(
            "hard.react.existing_project_not_new_app",
            document,
            "add to current product instead of scaffolding a new app",
            "Find evidence for adding React to an existing project, not creating a new app.",
            ("existing project current app add React incrementally",),
            (existing,),
            forbidden=(new_app,),
            near_miss=(new_app,),
            facets=(EvidenceFacet("existing", ("Existing Project",)),),
            min_support_score=0.40,
        ),
    )


def _react_adversarial_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    stack = _span(document, "does not make assumptions about the rest")
    rendering = _span(document, "render just the right components")
    component = _span(document, "Build encapsulated components")
    return (
        _case(
            "adversarial.react.stack_required",
            document,
            "requires a specific technology stack",
            "Find evidence that React requires a specific surrounding technology stack.",
            ("React requires a prescribed technology stack",),
            (),
            forbidden=(stack,),
            contradiction=(stack,),
            facets=(EvidenceFacet("stack", ("technology stack",)),),
            expect_abstain=True,
            min_support_score=0.55,
        ),
        _case(
            "adversarial.react_declarative_not_components",
            document,
            "state changes trigger targeted UI rendering",
            "Find evidence that React updates and renders the right components when data changes.",
            ("data changes efficiently update render right components",),
            (rendering,),
            forbidden=(component,),
            near_miss=(component,),
            facets=(
                EvidenceFacet("data", ("data changes",)),
                EvidenceFacet("render", ("render", "right components")),
            ),
        ),
    )


def _kubernetes_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    manage = _span(document, "managing containerized applications", kind=SpanKind.PARAGRAPH)
    cncf = _span(document, "hosted by the Cloud Native Computing Foundation")
    return (
        _case(
            "kubernetes.container_orchestration",
            document,
            "containerized application orchestration",
            "Find evidence that Kubernetes manages deployment and scaling across hosts.",
            ("managing containerized applications across hosts deployment scaling",),
            (manage,),
            facets=(
                EvidenceFacet("containerized", ("containerized applications",)),
                EvidenceFacet("scaling", ("deployment maintenance scaling",)),
            ),
        ),
        _case(
            "kubernetes.cncf",
            document,
            "foundation hosting",
            "Find evidence that Kubernetes is hosted by CNCF.",
            ("hosted by Cloud Native Computing Foundation",),
            (cncf,),
            facets=(EvidenceFacet("cncf", ("Cloud Native Computing Foundation", "CNCF")),),
        ),
    )


def _kubernetes_hard_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    borg = _span(document, "system called Borg")
    cncf = _span(document, "hosted by the Cloud Native Computing Foundation")
    using = _span(document, "See our documentation")
    return (
        _case(
            "hard.kubernetes.borg_lineage",
            document,
            "operational lineage from Borg at Google",
            "Find evidence that Kubernetes builds on Google's production Borg experience.",
            ("Google production workloads Borg experience",),
            (borg,),
            forbidden=(cncf,),
            facets=(
                EvidenceFacet("google", ("Google",)),
                EvidenceFacet("borg", ("Borg",)),
            ),
        ),
        _case(
            "hard.kubernetes.no_install_command",
            document,
            "kubectl install command",
            "Find evidence for a concrete kubectl install command.",
            ("kubectl install command package manager",),
            (),
            forbidden=(using,),
            insufficient_context=(using,),
            facets=(EvidenceFacet("kubectl", ("kubectl",)),),
            expect_abstain=True,
            min_support_score=0.55,
        ),
    )


def _kubernetes_adversarial_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    borg = _span(document, "system called Borg")
    cncf = _span(document, "hosted by the Cloud Native Computing Foundation")
    scaling = _span(document, "basic mechanisms for deployment")
    return (
        _case(
            "adversarial.kubernetes_hosted_by_google",
            document,
            "hosted by Google",
            "Find evidence that Kubernetes is hosted by Google.",
            ("hosted by Google",),
            (),
            forbidden=(borg, cncf),
            facets=(EvidenceFacet("google hosting", ("hosted", "Google")),),
            expect_abstain=True,
            min_support_score=0.55,
        ),
        _case(
            "adversarial.kubernetes_scaling_not_foundation",
            document,
            "scales deployed applications across hosts",
            "Find evidence for deployment, maintenance, and scaling across multiple hosts.",
            ("deployment maintenance scaling applications multiple hosts",),
            (scaling,),
            forbidden=(cncf,),
            facets=(
                EvidenceFacet("deployment", ("deployment",)),
                EvidenceFacet("scaling", ("scaling",)),
            ),
        ),
    )


def _django_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    framework = _span(document, "high-level Python web framework")
    install = _span(document, "docs/intro/install.txt")
    deployment = _span(document, "deployment server")
    return (
        _case(
            "django.framework",
            document,
            "rapid pragmatic Python web framework",
            "Find evidence that Django is a high-level Python web framework.",
            ("high-level Python web framework rapid development pragmatic design",),
            (framework,),
            facets=(EvidenceFacet("python web", ("Python web framework",)),),
        ),
        _case(
            "django.install_first",
            document,
            "first installation reading path",
            "Find evidence for which Django docs to read first when getting started.",
            ("first read docs intro install instructions",),
            (install,),
            forbidden=(deployment,),
            facets=(EvidenceFacet("install", ("docs/intro/install.txt", "installing Django")),),
        ),
    )


def _django_hard_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    ticket = _span(document, "fill out a")
    install = _span(document, "docs/intro/install.txt")
    tutorial = _span(document, "docs/intro/tutorial01.txt")
    return (
        _case(
            "hard.django.docs_bug_reporting",
            document,
            "report documentation defects",
            "Find evidence for how a user reports problems found in Django docs.",
            ("documentation problems report ticket code.djangoproject.com newticket",),
            (ticket,),
            forbidden=(install,),
            facets=(
                EvidenceFacet("problems", ("problems in the docs",)),
                EvidenceFacet("ticket", ("ticket",)),
            ),
        ),
        _case(
            "hard.django.tutorial_sequence",
            document,
            "tutorial order after installation",
            "Find evidence that the Django tutorials should be worked through in order.",
            ("tutorials in order tutorial01 tutorial02",),
            (tutorial,),
            forbidden=(install,),
            facets=(
                EvidenceFacet("order", ("in order",)),
                EvidenceFacet("tutorials", ("tutorial01", "tutorial02")),
            ),
        ),
    )


def _django_adversarial_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    install = _span(document, "docs/intro/install.txt")
    deployment = _span(document, "deployment server")
    ticket = _span(document, "fill out a")
    return (
        _case(
            "adversarial.django_first_doc_not_deployment",
            document,
            "first document for new users",
            "Find evidence for the first Django document to read when getting started.",
            ("just getting started first read install instructions",),
            (install,),
            forbidden=(deployment,),
            facets=(EvidenceFacet("first install", ("first read", "install")),),
        ),
        _case(
            "adversarial.django_no_chat_support",
            document,
            "chat support for docs issues",
            "Find evidence that Django docs problems should be reported through chat support.",
            ("docs problems chat support",),
            (),
            forbidden=(ticket,),
            facets=(EvidenceFacet("chat", ("chat support",)),),
            expect_abstain=True,
            min_support_score=0.55,
        ),
    )


def _fastapi_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    performance = _span(document, "Very high performance", kind=SpanKind.BULLET)
    standards = _span(document, "OpenAPI and JSON Schema")
    bugs = _span(document, "Reduce about 40%")
    return (
        _case(
            "fastapi.performance",
            document,
            "performance comparable to NodeJS and Go",
            "Find evidence that FastAPI is very high performance compared with NodeJS and Go.",
            ("very high performance on par with NodeJS and Go",),
            (performance,),
            facets=(
                EvidenceFacet("performance", ("Very high performance",)),
                EvidenceFacet("node go", ("NodeJS", "Go")),
            ),
        ),
        _case(
            "fastapi.openapi",
            document,
            "open standards API compatibility",
            "Find evidence that FastAPI is compatible with OpenAPI and JSON Schema.",
            ("compatible with open standards OpenAPI JSON Schema",),
            (standards,),
            forbidden=(bugs,),
            facets=(EvidenceFacet("standards", ("OpenAPI", "JSON Schema")),),
        ),
    )


def _fastapi_hard_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    interactive_docs = _span(document, "automatic interactive documentation")
    standards = _span(document, "OpenAPI and JSON Schema")
    easy = _span(document, "easy to learn")
    return (
        _case(
            "hard.fastapi.interactive_docs",
            document,
            "automatic interactive API documentation",
            "Find evidence that production-ready FastAPI code includes interactive docs.",
            ("automatic interactive documentation production-ready code",),
            (interactive_docs,),
            forbidden=(standards,),
            facets=(
                EvidenceFacet("interactive", ("interactive documentation",)),
                EvidenceFacet("production", ("production-ready",)),
            ),
        ),
        _case(
            "hard.fastapi.no_graphql_claim",
            document,
            "GraphQL schema generation",
            "Find evidence that FastAPI generates GraphQL schemas.",
            ("GraphQL schema generation",),
            (),
            forbidden=(standards, easy),
            insufficient_context=(standards, easy),
            facets=(EvidenceFacet("graphql", ("GraphQL",)),),
            expect_abstain=True,
            min_support_score=0.55,
        ),
    )


def _fastapi_adversarial_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    standards = _span(document, "OpenAPI and JSON Schema")
    performance = _span(document, "Very high performance", kind=SpanKind.BULLET)
    docs = _span(document, "automatic interactive documentation")
    return (
        _case(
            "adversarial.fastapi_openapi_without_json_schema",
            document,
            "OpenAPI support excluding JSON Schema",
            "Find evidence that FastAPI supports OpenAPI but not JSON Schema.",
            ("OpenAPI without JSON Schema",),
            (),
            forbidden=(standards,),
            contradiction=(standards,),
            facets=(EvidenceFacet("openapi", ("OpenAPI",)),),
            excluded_facets=(EvidenceFacet("json schema", ("JSON Schema",)),),
            expect_abstain=True,
            min_support_score=0.55,
        ),
        _case(
            "adversarial.fastapi_docs_not_speed",
            document,
            "automatic docs in production-ready code",
            "Find evidence for automatic interactive docs, not runtime performance.",
            ("automatic interactive documentation production-ready",),
            (docs,),
            forbidden=(performance,),
            near_miss=(performance,),
            facets=(EvidenceFacet("docs", ("interactive documentation",)),),
        ),
    )


def _pytorch_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    gpu = _span(document, "strong GPU acceleration")
    autograd = _span(document, "tape-based autograd")
    packages = _span(document, "NumPy, SciPy, and Cython")
    return (
        _case(
            "pytorch.gpu_tensors",
            document,
            "tensor computation with GPU acceleration",
            "Find evidence that PyTorch offers tensor computation with strong GPU acceleration.",
            ("tensor computation strong GPU acceleration",),
            (gpu,),
            facets=(EvidenceFacet("gpu", ("GPU acceleration",)),),
        ),
        _case(
            "pytorch.autograd",
            document,
            "tape based autograd neural networks",
            "Find evidence that PyTorch supports deep neural networks through tape-based autograd.",
            ("deep neural networks tape-based autograd",),
            (autograd,),
            forbidden=(gpu,),
            facets=(EvidenceFacet("autograd", ("tape-based autograd",)),),
        ),
        _case(
            "pytorch.python_extension",
            document,
            "extend with existing Python packages",
            "Find evidence that PyTorch can be extended with NumPy, SciPy, and Cython.",
            ("reuse NumPy SciPy Cython to extend PyTorch",),
            (packages,),
            facets=(EvidenceFacet("packages", ("NumPy", "SciPy", "Cython")),),
        ),
    )


def _pytorch_hard_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    ci = _span(document, "hud.pytorch.org")
    packages = _span(document, "NumPy, SciPy, and Cython")
    gpu = _span(document, "strong GPU acceleration")
    return (
        _case(
            "hard.pytorch.trunk_health_location",
            document,
            "where trunk health signals live",
            "Find evidence for where PyTorch publishes trunk health and CI signals.",
            ("trunk health continuous integration signals hud.pytorch.org",),
            (ci,),
            forbidden=(packages,),
            facets=(
                EvidenceFacet("trunk", ("trunk health",)),
                EvidenceFacet("ci", ("continuous integration", "hud.pytorch.org")),
            ),
        ),
        _case(
            "hard.pytorch.no_distributed_training",
            document,
            "distributed training launcher",
            "Find evidence for a distributed training launch command.",
            ("distributed training launcher torchrun command",),
            (),
            forbidden=(gpu,),
            facets=(EvidenceFacet("distributed", ("distributed", "torchrun")),),
            expect_abstain=True,
            min_support_score=0.55,
        ),
    )


def _pytorch_adversarial_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    gpu = _span(document, "strong GPU acceleration")
    autograd = _span(document, "tape-based autograd")
    packages = _span(document, "NumPy, SciPy, and Cython")
    return (
        _case(
            "adversarial.pytorch_cpu_only_tensors",
            document,
            "CPU-only tensor computation",
            "Find evidence that PyTorch tensor computation is CPU-only.",
            ("tensor computation CPU only no GPU acceleration",),
            (),
            forbidden=(gpu,),
            contradiction=(gpu,),
            facets=(EvidenceFacet("tensor", ("tensor computation",)),),
            excluded_facets=(EvidenceFacet("gpu", ("GPU acceleration",)),),
            expect_abstain=True,
            min_support_score=0.55,
        ),
        _case(
            "adversarial.pytorch_autograd_not_packages",
            document,
            "neural network differentiation engine",
            "Find evidence for PyTorch's tape-based autograd, not extension packages.",
            ("deep neural networks tape-based autograd",),
            (autograd,),
            forbidden=(packages,),
            facets=(EvidenceFacet("autograd", ("tape-based autograd",)),),
        ),
    )


def _rust_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    repo = _span(document, "contains the compiler")
    reliability = _span(document, "ownership model ensure memory")
    productivity = _span(document, "Cargo, rustfmt, Clippy")
    return (
        _case(
            "rust.repository_contents",
            document,
            "compiler standard library documentation repository",
            "Find evidence for what the Rust repository contains.",
            ("compiler standard library documentation",),
            (repo,),
            facets=(EvidenceFacet("contents", ("compiler", "standard library", "documentation")),),
        ),
        _case(
            "rust.memory_thread_safety",
            document,
            "ownership model memory and thread safety",
            "Find evidence that Rust's ownership model supports memory and thread safety.",
            ("ownership model memory and thread safety reducing bugs",),
            (reliability,),
            facets=(EvidenceFacet("safety", ("memory", "thread safety", "ownership model")),),
        ),
        _case(
            "rust.tooling",
            document,
            "Cargo and linting tools",
            "Find evidence that Rust has advanced tooling including Cargo and Clippy.",
            ("Cargo rustfmt Clippy rust-analyzer advanced tooling",),
            (productivity,),
            facets=(EvidenceFacet("tools", ("Cargo", "Clippy", "rust-analyzer")),),
        ),
    )


def _rust_hard_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    diagnostics = _span(document, "compiler committed to great")
    reliability = _span(document, "ownership model ensure memory")
    repo = _span(document, "contains the compiler")
    return (
        _case(
            "hard.rust.compiler_diagnostics",
            document,
            "compiler diagnostics as productivity feature",
            "Find evidence that Rust productivity includes compiler diagnostics and tooling.",
            ("compiler diagnostics advanced tooling productivity",),
            (diagnostics,),
            forbidden=(reliability,),
            facets=(
                EvidenceFacet("diagnostics", ("diagnostics",)),
                EvidenceFacet("tooling", ("Cargo", "Clippy", "rust-analyzer")),
            ),
        ),
        _case(
            "hard.rust.no_package_registry",
            document,
            "package registry publishing",
            "Find evidence for publishing a Rust crate to a package registry.",
            ("publish crate package registry crates.io",),
            (),
            forbidden=(repo,),
            facets=(EvidenceFacet("registry", ("crates.io", "package registry")),),
            expect_abstain=True,
            min_support_score=0.55,
        ),
    )


def _rust_adversarial_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    reliability = _span(document, "ownership model ensure memory")
    diagnostics = _span(document, "compiler committed to great")
    repo = _span(document, "contains the compiler")
    return (
        _case(
            "adversarial.rust_runtime_safety",
            document,
            "runtime memory safety checks",
            "Find evidence that Rust relies on runtime checks for memory and thread safety.",
            ("runtime checks memory thread safety",),
            (),
            forbidden=(reliability,),
            contradiction=(reliability,),
            facets=(EvidenceFacet("safety", ("memory", "thread safety")),),
            excluded_facets=(EvidenceFacet("compile time", ("compile-time",)),),
            expect_abstain=True,
            min_support_score=0.55,
        ),
        _case(
            "adversarial.rust_diagnostics_not_repo_contents",
            document,
            "compiler diagnostics and tooling productivity",
            "Find evidence for productivity from diagnostics and tools, not repository contents.",
            ("compiler diagnostics Cargo Clippy rust-analyzer productivity",),
            (diagnostics,),
            forbidden=(repo,),
            facets=(
                EvidenceFacet("diagnostics", ("diagnostics",)),
                EvidenceFacet("tooling", ("Cargo", "Clippy")),
            ),
        ),
    )


def _vscode_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    oss = _span(document, "develops the Visual Studio Code product together")
    distribution = _span(document, "Microsoft-specific customizations")
    cycle = _span(document, "edit-build-debug cycle")
    docs = _span(document, "Review the documentation")
    return (
        _case(
            "vscode.code_oss_repository",
            document,
            "community development repository",
            "Find evidence that Code - OSS is where VS Code is developed with the community.",
            ("develop product together with the community",),
            (oss,),
            facets=(EvidenceFacet("community", ("together with the community",)),),
        ),
        _case(
            "vscode.product_license",
            document,
            "distribution customizations product license",
            "Find evidence that Visual Studio Code distribution has Microsoft customizations.",
            ("distribution Code OSS Microsoft-specific customizations product license",),
            (distribution,),
            facets=(EvidenceFacet("customizations", ("Microsoft-specific customizations",)),),
        ),
        _case(
            "vscode.editor_cycle",
            document,
            "editor core edit build debug loop",
            "Find evidence that VS Code combines editor simplicity with edit-build-debug work.",
            ("simplicity code editor core edit-build-debug cycle",),
            (cycle,),
            facets=(EvidenceFacet("cycle", ("edit-build-debug cycle",)),),
        ),
        _case(
            "vscode.docs_contribution",
            document,
            "documentation pull requests",
            "Find evidence that contributors can review docs and open pull requests.",
            ("review documentation make pull requests",),
            (docs,),
            facets=(EvidenceFacet("docs", ("Review the documentation", "pull requests")),),
        ),
    )


def _vscode_hard_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    verify = _span(document, "help verify them")
    docs = _span(document, "Review the documentation")
    distribution = _span(document, "Microsoft-specific customizations")
    return (
        _case(
            "hard.vscode.verify_bugs",
            document,
            "verify submitted bugs and feature requests",
            "Find evidence that contributors can help verify bugs and feature requests.",
            ("submit bugs feature requests help verify checked in",),
            (verify,),
            forbidden=(docs,),
            facets=(
                EvidenceFacet("bugs", ("bugs", "feature requests")),
                EvidenceFacet("verify", ("verify",)),
            ),
        ),
        _case(
            "hard.vscode.no_extension_marketplace",
            document,
            "extension marketplace publishing",
            "Find evidence for publishing an extension to the VS Code marketplace.",
            ("extension marketplace publishing",),
            (),
            forbidden=(distribution,),
            facets=(EvidenceFacet("marketplace", ("marketplace", "extension")),),
            expect_abstain=True,
            min_support_score=0.55,
        ),
    )


def _vscode_adversarial_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    distribution = _span(document, "Microsoft-specific customizations")
    oss = _span(document, "develops the Visual Studio Code product together")
    cycle = _span(document, "edit-build-debug cycle")
    return (
        _case(
            "adversarial.vscode_code_oss_is_product_distribution",
            document,
            "Code OSS includes product license customizations",
            (
                "Find evidence that Code - OSS itself includes Microsoft product "
                "license customizations."
            ),
            ("Code OSS Microsoft-specific customizations product license",),
            (),
            forbidden=(distribution, oss),
            facets=(EvidenceFacet("customizations", ("Microsoft-specific customizations",)),),
            expect_abstain=True,
            min_support_score=0.55,
        ),
        _case(
            "adversarial.vscode_editor_cycle_not_distribution",
            document,
            "editor supports edit build debug loop",
            "Find evidence that VS Code supports the edit-build-debug workflow.",
            ("code editor edit-build-debug cycle",),
            (cycle,),
            forbidden=(distribution,),
            facets=(EvidenceFacet("cycle", ("edit-build-debug cycle",)),),
        ),
    )


def _ansible_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    automation = _span(document, "configuration management")
    rolling = _span(document, "zero-downtime rolling updates")
    agentless = _span(document, "agentless by leveraging")
    non_python = _span(document, "not just Python")
    return (
        _case(
            "ansible.automation_scope",
            document,
            "IT automation scope",
            "Find evidence that Ansible handles configuration, deployment, and orchestration.",
            ("configuration management application deployment orchestration automation",),
            (automation,),
            facets=(
                EvidenceFacet("configuration", ("configuration management",)),
                EvidenceFacet("orchestration", ("multi-node orchestration",)),
            ),
        ),
        _case(
            "ansible.rolling_updates",
            document,
            "zero downtime load balancer updates",
            "Find evidence that Ansible supports zero-downtime rolling updates.",
            ("zero-downtime rolling updates load balancers",),
            (rolling,),
            facets=(EvidenceFacet("rolling", ("zero-downtime rolling updates",)),),
        ),
        _case(
            "ansible.agentless",
            document,
            "agentless through SSH",
            "Find evidence that Ansible avoids custom agents and uses SSH.",
            ("agentless existing SSH daemon avoid custom agents",),
            (agentless,),
            facets=(EvidenceFacet("ssh", ("agentless", "SSH daemon")),),
        ),
        _case(
            "ansible.dynamic_language",
            document,
            "module development not limited to Python",
            (
                "Find evidence that Ansible modules can be developed in dynamic languages "
                "beyond Python."
            ),
            ("module development any dynamic language not just Python",),
            (non_python,),
            facets=(EvidenceFacet("language", ("dynamic language", "not just Python")),),
        ),
    )


def _ansible_hard_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    human = _span(document, "machine and human friendly")
    security = _span(document, "security and easy auditability")
    remote = _span(document, "without bootstrapping")
    return (
        _case(
            "hard.ansible.human_friendly_infra",
            document,
            "human-readable infrastructure language",
            "Find evidence that Ansible describes infrastructure in a human-friendly language.",
            ("infrastructure language machine human friendly",),
            (human,),
            forbidden=(security,),
            facets=(
                EvidenceFacet("infrastructure", ("infrastructure",)),
                EvidenceFacet("human", ("human friendly",)),
            ),
        ),
        _case(
            "hard.ansible.instant_remote_management",
            document,
            "manage remote machines without bootstrap software",
            "Find evidence that Ansible can manage new remote machines without bootstrapping.",
            ("new remote machines instantly without bootstrapping software",),
            (remote,),
            forbidden=(human,),
            facets=(
                EvidenceFacet("remote", ("remote machines",)),
                EvidenceFacet("bootstrap", ("without bootstrapping",)),
            ),
        ),
    )


def _ansible_adversarial_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    agentless = _span(document, "agentless by leveraging")
    automation = _span(document, "configuration management")
    remote = _span(document, "without bootstrapping")
    return (
        _case(
            "adversarial.ansible_requires_custom_agents",
            document,
            "requires custom agents",
            "Find evidence that Ansible requires custom agents and extra open ports.",
            ("requires custom agents additional open ports",),
            (),
            forbidden=(agentless,),
            contradiction=(agentless,),
            facets=(EvidenceFacet("agents", ("custom agents", "open ports")),),
            excluded_facets=(EvidenceFacet("agentless", ("agentless", "avoid custom agents")),),
            expect_abstain=True,
            min_support_score=0.55,
        ),
        _case(
            "adversarial.ansible_remote_bootstrap_not_scope",
            document,
            "remote machines need no bootstrap software",
            "Find evidence that new remote machines can be managed without bootstrapping.",
            ("new remote machines without bootstrapping software",),
            (remote,),
            forbidden=(automation,),
            facets=(
                EvidenceFacet("remote", ("remote machines",)),
                EvidenceFacet("bootstrap", ("without bootstrapping",)),
            ),
        ),
    )


def _case(
    case_id: str,
    document: EvidenceDocument,
    label: str,
    description: str,
    positive_examples: tuple[str, ...],
    support: tuple[DocumentSpan, ...],
    *,
    negative_examples: tuple[str, ...] = (),
    near_miss: tuple[DocumentSpan, ...] = (),
    contradiction: tuple[DocumentSpan, ...] = (),
    insufficient_context: tuple[DocumentSpan, ...] = (),
    forbidden: tuple[DocumentSpan, ...] = (),
    facets: tuple[EvidenceFacet, ...] = (),
    excluded_facets: tuple[EvidenceFacet, ...] = (),
    relations: tuple[EvidenceRelation, ...] = (),
    expect_abstain: bool = False,
    min_support_score: float = 0.42,
    difficulty: str | None = None,
    phenomena: tuple[str, ...] = (),
) -> BenchmarkCase:
    return BenchmarkCase(
        id=case_id,
        intent=IntentSpec(
            id=case_id,
            label=label,
            description=description,
            positive_examples=positive_examples,
            negative_examples=negative_examples,
            required_facets=facets,
            excluded_facets=excluded_facets,
            relations=relations,
            min_support_score=min_support_score,
        ),
        document=document,
        support_span_ids=tuple(span.id for span in support),
        near_miss_span_ids=tuple(span.id for span in near_miss),
        contradiction_span_ids=tuple(span.id for span in contradiction),
        insufficient_context_span_ids=tuple(span.id for span in insufficient_context),
        forbidden_span_ids=tuple(span.id for span in forbidden),
        expect_abstain=expect_abstain,
        curation=_curation_for_case(
            support=support,
            near_miss=near_miss,
            contradiction=contradiction,
            insufficient_context=insufficient_context,
            forbidden=forbidden,
            excluded_facets=excluded_facets,
            relations=relations,
            expect_abstain=expect_abstain,
            difficulty=difficulty,
            extra_phenomena=phenomena,
        ),
    )


def _curation_for_case(
    *,
    support: tuple[DocumentSpan, ...],
    near_miss: tuple[DocumentSpan, ...],
    contradiction: tuple[DocumentSpan, ...],
    insufficient_context: tuple[DocumentSpan, ...],
    forbidden: tuple[DocumentSpan, ...],
    excluded_facets: tuple[EvidenceFacet, ...],
    relations: tuple[EvidenceRelation, ...],
    expect_abstain: bool,
    difficulty: str | None,
    extra_phenomena: tuple[str, ...],
) -> BenchmarkCuration:
    phenomena = (
        *extra_phenomena,
        *(() if not support else ("direct_support",)),
        *(() if not expect_abstain else ("abstention",)),
        *(() if not forbidden else ("hard_negative",)),
        *(() if not near_miss else ("near_miss",)),
        *(() if not contradiction else ("contradiction",)),
        *(() if not insufficient_context else ("insufficient_context",)),
        *(() if not excluded_facets else ("excluded_facet",)),
        *(() if not relations else ("relation_binding",)),
    )
    return BenchmarkCuration(
        reviewed=True,
        source="curated_readme_excerpt",
        difficulty=difficulty
        or _difficulty_for_case(
            expect_abstain=expect_abstain,
            forbidden=forbidden,
            near_miss=near_miss,
            contradiction=contradiction,
            insufficient_context=insufficient_context,
            relations=relations,
        ),
        phenomena=_dedupe_strings(phenomena),
    )


def _difficulty_for_case(
    *,
    expect_abstain: bool,
    forbidden: tuple[DocumentSpan, ...],
    near_miss: tuple[DocumentSpan, ...],
    contradiction: tuple[DocumentSpan, ...],
    insufficient_context: tuple[DocumentSpan, ...],
    relations: tuple[EvidenceRelation, ...],
) -> str:
    if relations or contradiction or insufficient_context:
        return "hard"
    if expect_abstain or forbidden or near_miss:
        return "medium"
    return "standard"


def _dedupe_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return tuple(deduped)


def _span(
    document: EvidenceDocument,
    text: str,
    *,
    kind: SpanKind | None = None,
) -> DocumentSpan:
    matches = [
        span
        for span in document.spans
        if text in span.text and (kind is None or span.kind == kind)
    ]
    if not matches:
        raise ValueError(f"no span in {document.id!r} contains {text!r}")
    return min(matches, key=lambda span: len(span.text))
