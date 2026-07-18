# RepoPilot Backend Phase 1 - Known Limitations

1. **Size Limits**: The GitHub API integration currently caps processing at `51200KB` (50MB) repository size or 300 files. Larger repos are blocked to avoid rate limits and slow processing times.
2. **Rate Limits**: The backend sequentializes file fetching via `httpx`. Heavy usage on a single IP might trigger GitHub REST API rate limits.
3. **Language Chunking**: We use `tree-sitter` for code chunking, but fallback to naive text splitting if a language's grammar isn't loaded or recognized. Only a few top languages currently have high-fidelity chunking boundaries configured.
4. **Model Compatibility**: Currently optimized for `openai/gpt-oss-120b` (via NVIDIA) and Groq models. Prompts might need tuning if switching to smaller local models.
5. **Context Window Limitations**: Repositories with extremely large entry points or architecture documents might get truncated when constructing the final summary context.
