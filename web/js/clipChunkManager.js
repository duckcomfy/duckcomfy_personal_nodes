import { app } from "/scripts/app.js";

app.registerExtension({
    name: "DuckComfy.ClipChunkManager",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ClipChunkManager") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                const textWidget = this.widgets.find((w) => w.name === "text");
                if (!textWidget?.inputEl) return;
                textWidget.inputEl.addEventListener("blur", async (event) => {
                    const prompt = event.target.value;
                    const clipInput = this.inputs?.find(
                        (input) => input.name === "clip"
                    );
                    console.log(this.inputs);

                    let checkpointName = null;
                    let sourceNode = null;

                    // First try to find via direct link
                    if (clipInput?.link) {
                        // Get the link object from the graph
                        const linkObj = this.graph.links[clipInput.link];
                        sourceNode = linkObj
                            ? this.graph.getNodeById(linkObj.origin_id)
                            : null;

                        if (
                            sourceNode &&
                            sourceNode.type === "CheckpointLoaderSimple"
                        ) {
                            // Get the checkpoint filename from the widget
                            const ckptWidget = sourceNode.widgets?.find(
                                (w) => w.name === "ckpt_name"
                            );
                            if (ckptWidget) {
                                checkpointName = ckptWidget.value;
                            }
                        }
                    }

                    // Fallback: If no checkpoint found via direct link, search entire workflow
                    if (!checkpointName) {
                        console.log(
                            "No direct CLIP link found, searching for any CheckpointLoader in workflow..."
                        );

                        // Search all nodes for CheckpointLoaderSimple
                        for (const node of this.graph._nodes) {
                            if (
                                node.type === "CheckpointLoaderSimple" ||
                                node.type === "CheckpointLoader"
                            ) {
                                const ckptWidget = node.widgets?.find(
                                    (w) => w.name === "ckpt_name"
                                );
                                if (ckptWidget?.value) {
                                    checkpointName = ckptWidget.value;
                                    break;
                                }
                            }
                        }
                    }

                    if (!checkpointName) {
                        console.warn(
                            "Couldn't auto-add BREAK keywords at CLIP chunk boundaries because no checkpoint could be found. " +
                                "Please ensure there's a CheckpointLoader node in the workflow."
                        );
                        return;
                    }
                    const resp = await fetch("/api/query_tokenized_chunks", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json",
                        },
                        body: JSON.stringify({
                            text: prompt,
                            checkpoint_name: checkpointName,
                        }),
                    }).then((r) => r.json());
                    console.log(resp);

                    const newPrompt = resp.prompt_with_breaks;
                    if (newPrompt) {
                        textWidget.inputEl.value = newPrompt;
                    }
                });
            };
        }
    },
});

const splitAtIndices = (arr, indices) => {
    if (indices.length === 0) return [arr];
    const sortedIndices = [...indices].sort((a, b) => a - b);
    const allIndices = [0, ...sortedIndices, arr.length];
    return allIndices
        .slice(0, -1)
        .map((start, i) => arr.slice(start, allIndices[i + 1]))
        .filter((subArr) => subArr.length > 0);
};
