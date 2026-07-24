## GitHub Copilot Chat

- Extension: 0.37.9 (prod)
- VS Code: 1.109.5 (072586267e68ece9a47aa43f8c108e0dcbf44622)
- OS: win32 10.0.26200 x64
- GitHub Account: emmorais

## Network

User Settings:
```json
  "http.systemCertificatesNode": true,
  "github.copilot.advanced.debug.useElectronFetcher": true,
  "github.copilot.advanced.debug.useNodeFetcher": false,
  "github.copilot.advanced.debug.useNodeFetchFetcher": true
```

Connecting to https://api.github.com:
- DNS ipv4 Lookup: 4.228.31.149 (43 ms)
- DNS ipv6 Lookup: Error (7 ms): getaddrinfo ENOTFOUND api.github.com
- Proxy URL: None (1 ms)
- Electron fetch (configured): timed out after 10 seconds
- Node.js https: timed out after 10 seconds
- Node.js fetch: timed out after 10 seconds

Connecting to https://api.githubcopilot.com/_ping:
- DNS ipv4 Lookup: 140.82.112.22 (162 ms)
- DNS ipv6 Lookup: Error (6 ms): getaddrinfo ENOTFOUND api.githubcopilot.com
- Proxy URL: None (2 ms)
- Electron fetch (configured): HTTP 200 (745 ms)
- Node.js https: HTTP 200 (794 ms)
- Node.js fetch: HTTP 200 (785 ms)

Connecting to https://copilot-proxy.githubusercontent.com/_ping:
- DNS ipv4 Lookup: 4.228.31.153 (37 ms)
- DNS ipv6 Lookup: Error (30 ms): getaddrinfo ENOTFOUND copilot-proxy.githubusercontent.com
- Proxy URL: None (3 ms)
- Electron fetch (configured): HTTP 200 (259 ms)
- Node.js https: HTTP 200 (232 ms)
- Node.js fetch: HTTP 200 (231 ms)

Connecting to https://mobile.events.data.microsoft.com: HTTP 404 (1294 ms)
Connecting to https://dc.services.visualstudio.com: HTTP 404 (1167 ms)
Connecting to https://copilot-telemetry.githubusercontent.com/_ping: HTTP 200 (619 ms)
Connecting to https://copilot-telemetry.githubusercontent.com/_ping: HTTP 200 (623 ms)
Connecting to https://default.exp-tas.com: HTTP 400 (217 ms)

Number of system certificates: 99

## Documentation

In corporate networks: [Troubleshooting firewall settings for GitHub Copilot](https://docs.github.com/en/copilot/troubleshooting-github-copilot/troubleshooting-firewall-settings-for-github-copilot).