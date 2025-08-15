---
id: "parameter-tying-framework"
title: "Design parameter tying framework for GPy multioutput kernels"
status: "Ready"
priority: "High"
created: "2025-08-15"
last_updated: "2025-08-15"
owner: "Neil Lawrence"
github_issue: ""
dependencies: ""
tags:
- parameter-tying
- multioutput
- kernel-framework
- architecture
---

# Investigate parameter tying limitations and create CIP for discussion

## Description

During LFM kernel code review, we identified that GPy lacks systematic parameter tying capabilities compared to GPmat's `modelTieParam()` functionality. This limitation affects combination kernels such as multiouptut or additive kernels. We need to investigate the scope of this problem and create a CIP to discuss potential solutions with the community.

## Problem Statement

- **Current Limitation**: GPy's parameter system doesn't support tying parameters across different kernel components
- **Impact on LFM**: Forces complex parameter handling in EQ_ODE1 and EQ_ODE2 kernels
- **Broader Impact**: May affect other multiple kernel scenarios where parameters should be shared
- **Comparison**: GPmat has `modelTieParam()` functionality that GPy lacks

## Investigation Needed

### 1. Scope Assessment
- [x] Search existing GitHub issues for parameter tying discussions
- [ ] Identify other kernels/models that could benefit from parameter tying
- [ ] Assess impact on current GPy codebase


### 2. Community Input
- [ ] Create GitHub issue and associated CIP to discuss parameter tying needs
- [ ] Gather feedback from GPy maintainers and users
- [ ] Identify use cases beyond LFM kernels
- [ ] Assess priority relative to other GPy improvements

### 3. Technical Analysis
- [ ] Analyze GPmat's parameter tying implementation
- [ ] Review GPy's current parameter system architecture
- [ ] Identify potential integration points
- [ ] Assess complexity and maintenance burden

## Acceptance Criteria

- [ ] Complete investigation of existing GitHub issues and discussions
- [ ] Document scope of parameter tying needs across GPy
- [ ] Create CIP for parameter tying framework discussion
- [ ] Gather community feedback on approach and priority
- [ ] Provide recommendations for next steps

## Implementation Notes

- Focus on problem identification and community discussion
- Avoid prescribing specific solutions until community input is gathered
- Consider whether this should be a separate CIP or part of broader multioutput improvements
- Document trade-offs between different approaches

## Related

- CIP: 0001 (LFM kernel implementation) - may depend on parameter tying
- GPy parameter system design
- GPmat parameter tying implementation
- Multioutput kernel architecture discussions

## Progress Updates

### 2025-08-15
Task created after identifying parameter tying as a potential limitation during LFM kernel code review. Need to investigate scope and create CIP for community discussion.

### 2025-08-15 (GitHub Investigation)
Found existing GitHub issues confirming parameter tying limitations:

**GPy Issues:**
- **Issue #462 (2016)**: "tie_params doesnt work ?" - `AttributeError: 'Add' object has no attribute 'tie_params'`
- **Issue #789 (2019)**: "Non-implemented Param tying work-around options" - Confirms `tie_to` from Parametrized is not implemented
- **Issue #878 (2020)**: "Constraining hyperparameters" - Open issue requesting parameter equality constraints in MultioutputGP

**Paramz Issues:**
- **Issue #34 (2019)**: "What does m.name[0].tie_to(other) do?" - `tie_to` is documented but not implemented
- **Issue #35 (2020)**: "Constraint that makes parameters sum to one?" - Also references missing `tie_to` functionality

**Key Findings:**
- Parameter tying functionality has been missing/broken in both GPy and paramz for at least 5 years
- `tie_to` method is documented in paramz but not implemented
- Multiple users have requested this feature for different use cases
- Current workarounds involve manual parameter management
- No systematic solution exists in either codebase
- The problem is deeper than just GPy - it's a fundamental limitation in the paramz framework

### 2025-08-15 (Paramz Dependency Analysis)
**Current State:**
- GPy still actively depends on paramz: `"paramz>=0.9.6"` in setup.py
- No evidence of plans to remove paramz dependency
- Recent paramz-related work: Issue #978 (2022) fixing broken kernels due to `add_parameter` → `link_parameter` rename

**Implications for Parameter Tying:**
- **Option 1**: Fix paramz first - implement `tie_to` in paramz framework
- **Option 2**: Work around paramz - create parameter tying within GPy without relying on paramz's missing features  
- **Option 3**: Replace paramz - major migration away from paramz (unlikely given current dependency)

**Recommendation**: Focus on Option 1 or 2, as paramz remains actively maintained and GPy continues to depend on it.
