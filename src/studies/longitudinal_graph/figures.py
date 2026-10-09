"""Separate topology, community and representation QC figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def mapper_figures(scenario, directory, dpi):
    graph=scenario["graph"]; nodes=scenario["node_table"].set_index("node_id"); summary=scenario["summary"]
    positions={n:np.mean(scenario["embedding"][list(m),:2],axis=0) for n,m in graph["nodes"].items()}
    supported_macros=set(scenario["macrostate_table"].loc[scenario["macrostate_table"].macrostate_supported,"macrostate_id"])
    for communities,filename in ((False,"02_reference_visit_mapper.png"),(True,"02_reference_visit_mapper_macrostates.png")):
        fig,ax=plt.subplots(figsize=(10,7)); legend=[]
        for a,b in scenario["raw_nerve"].edges:
            shared=len(graph["nodes"][a]&graph["nodes"][b])
            ax.plot([positions[a][0],positions[b][0]],[positions[a][1],positions[b][1]],color=".75",lw=.5+np.log1p(shared),zorder=1)
        identities={}
        for node,pos in positions.items():
            macro=scenario["node_to_macrostate"].get(node)
            identity=macro if communities else int(nodes.loc[node,"connected_component_id"])
            color=plt.get_cmap("tab10")(int(identity)%10) if identity is not None else ".65"
            supported=bool(nodes.loc[node,"supported"]) and (not communities or macro in supported_macros)
            ax.scatter(*pos,s=30+8*nodes.loc[node,"n_unique_patients"],c=[color],marker="o" if supported else "s",
                       edgecolors="black" if supported else "red",linewidths=1.3,zorder=2)
            identities[identity]=color
        for identity,color in identities.items():
            if communities and identity is not None:
                m=scenario["macrostate_table"].set_index("macrostate_id").loc[identity]
                label=f"Community {identity}: {m.n_unique_patients} patients ({'supported' if m.macrostate_supported else 'exploratory'})"
            else: label=f"Component {identity}" if identity is not None else "Unsupported node"
            legend.append(Line2D([0],[0],marker="o",color=color,ls="",label=label))
        legend += [Line2D([0],[0],marker="s",markeredgecolor="red",color=".65",ls="",label="Unsupported node or community")]
        ax.legend(handles=legend,loc="best",fontsize=8)
        ratios=scenario["pca"].explained_variance_ratio_
        annotation=(f"{summary['scenario']}; visits={summary['n_visits']}; patients={summary['n_patients']}; nodes={summary['n_nodes']}\n"
                    f"graph coverage={summary['pct_mapper_covered_visits']:.1f}%; supported visits={summary['n_supported_macrostate_covered_visits']}\n"
                    f"PC1/PC2 variance={100*ratios[0]:.1f}% / {100*ratios[1]:.1f}%")
        ax.set(title="Mapper communities / macrostates" if communities else "Visit Mapper connected components",xlabel="PC1 lens",ylabel="PC2 lens")
        fig.text(.1,.02,"edges = shared visits, not temporal transitions; width = 0.5 + log(1 + shared visits)",fontsize=9)
        fig.text(.1,.90,annotation,fontsize=9); fig.tight_layout(rect=(0,.06,1,.87))
        fig.savefig(directory/filename,dpi=dpi);plt.close(fig)


def diagnostic_figures(primary, tables, comparisons, bootstrap, directory, dpi):
    def save(fig,name): fig.tight_layout();fig.savefig(directory/name,dpi=dpi);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for pc,ax in enumerate(axes):
        x=primary["embedding"][:,pc];ax.hist(x,bins=30,color="steelblue")
        for q in np.quantile(x,[.01,.5,.99]):ax.axvline(q,color="darkred",lw=.8)
        ax.set(title=f"PC{pc+1}: original range",xlabel="Score",ylabel="Visits")
    save(fig,"02_lens_distributions.png")
    fig,ax=plt.subplots(figsize=(6,5));cover=primary["cover_qc"].pivot(index="cube_y",columns="cube_x",values="n_points")
    im=ax.imshow(cover,origin="lower",cmap="Blues");fig.colorbar(im,ax=ax,label="Visits (overlap permits repeats)")
    ax.set(title="Lens cover occupancy",xlabel="PC1 band",ylabel="PC2 band");save(fig,"02_lens_cube_occupancy.png")
    fig,ax=plt.subplots(figsize=(8,4));detail=tables["mapper_coverage_detail"]
    counts=detail.coverage_category.value_counts();counts.plot.bar(ax=ax,color="steelblue")
    ax.set(title="Exclusive visit coverage categories",ylabel="Visits");ax.tick_params(axis="x",rotation=25)
    save(fig,"02_mapper_coverage_breakdown.png")
    fig,axes=plt.subplots(2,2,figsize=(12,8));loading=tables["pca_loading_dominance"]
    for i,ax in enumerate(axes.flat):
        group=loading.loc[loading.scenario.eq("S3-primary")&loading.component.eq(f"PC{i+1}")].sort_values("squared_loading_fraction").tail(8)
        ax.barh(group.feature,group.squared_loading_fraction);ax.set(title=f"PC{i+1}: within-component loading fractions",xlabel="Squared loading / all squared loadings")
        ax.tick_params(axis="y",labelsize=7)
    save(fig,"02_pca_dominance.png")
    fig,ax=plt.subplots(figsize=(10,5));labels=[];values=[]
    for name,c in comparisons.items():
        if name=="S3-primary" or c.get("patient_jaccard_vs_primary") is None:continue
        labels.append(f"{name} (n={c.get('n_visits_compared',0)})");values.append(c["patient_jaccard_vs_primary"])
    ax.barh(labels,values);ax.set(xlim=(0,1),xlabel="Mean patient Jaccard; review comparability and individual structures")
    save(fig,"02_mapper_sensitivity_comparison.png")
    fig,ax=plt.subplots(figsize=(7,4))
    if len(bootstrap):ax.hist(bootstrap.n_nodes.dropna(),bins=15)
    else:ax.text(.5,.5,"Bootstrap not applicable: inspect summary",ha="center",transform=ax.transAxes)
    ax.set(xlabel="Nodes per patient refit",ylabel="Replicates");save(fig,"02_mapper_stability_distributions.png")
