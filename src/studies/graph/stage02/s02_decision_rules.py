"""Fixed technical release gates; never accepts clinical outcomes."""


def release_decision(*,primary_built,primary_qc,stable_count,mode,limitations,artifact_risk):
    if not primary_built or not primary_qc:
        return 'QC_FAILED',['PRIMARY_CONSTRUCTION_OR_RESAMPLING_QC_FAILED']
    if stable_count==0:
        return 'NO_STABLE_STRUCTURE',['No supported community passes prespecified repeated-subsampling stability']
    if mode=='exploratory' or limitations or artifact_risk:
        return 'PARTIAL_OR_EXPLORATORY',list(limitations)+(['Administrative/null controls require caution'] if artifact_risk else [])
    return 'VALIDATED_TOPOLOGICAL_STRUCTURE',[]
