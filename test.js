async function predict() {

    const spaceUrl =
        "https://vicemusic5-kaust-smile-predict.hf.space";


    // submit
    const submit = await fetch(
        `${spaceUrl}/gradio_api/call/predict`,
        {
            method: "POST",
            headers:{
                "Content-Type":"application/json"
            },
            body:JSON.stringify({
                data:[
                    ["CCO","CCN"],
                    "SIF",
                    "Morgan",
                    "lr"
                ]
            })
        }
    );


    const {event_id} = await submit.json();

    console.log(
        "event:",
        event_id
    );


    // listen SSE
    const result = await fetch(
        `${spaceUrl}/gradio_api/call/predict/${event_id}`
    );


    const reader =
        result.body.getReader();


    const decoder =
        new TextDecoder();


    while(true){

        const {
            done,
            value
        } = await reader.read();


        if(done)
            break;


        const chunk =
            decoder.decode(value);


        console.log(chunk);


        // Gradio最终返回：
        // data: {"msg":"process_completed","output":...}

        if(chunk.includes("process_completed")){

            const line =
                chunk
                .split("\n")
                .find(
                    x=>x.startsWith("data:")
                );


            const json =
                JSON.parse(
                    line.replace(
                        "data:",
                        ""
                    )
                );


            console.log(
                "FINAL:",
                json
            );

            break;
        }
    }
}


predict();

// 数据获取地址：https://huggingface.co/datasets/Vicemusic5/KAUST-SMILES-SIFSGF