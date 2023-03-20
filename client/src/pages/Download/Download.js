import React from "react";
import { useState, useEffect, useContext } from "react";
import Navbar from "../../components/nav/navbar/Navbar";
import Footer from "../../components/footer/Footer";
import { Link } from "react-router-dom";
import "./Download.css";
import plantUmlEncoder from "plantuml-encoder";
import DiagramMarkdownContext from "../../context/DiagramMarkdownContext";

const Download = () => {
    const { responseData } = useContext(DiagramMarkdownContext); // Context
    const [diagramUrl, setDiagramUrl] = useState("");

    // creating img from plantuml code
    const encodeDiagram = (plantUmlCode) => {
        console.log(plantUmlCode);
        const encodedCode = plantUmlEncoder.encode(plantUmlCode);
        const url = `http://www.plantuml.com/plantuml/img/${encodedCode}`;
        return url;
    };

    // run encodeDiagram fun on page load
    useEffect(() => {
        const encodedDiagram = encodeDiagram(responseData);
        setDiagramUrl(encodedDiagram);
    }, []);

    return (
        <div>
            <Navbar />
            <div>
                <div className="download-container">
                    <h2 className="mb-0">GenUML</h2>
                    <p>Final output of your use case diagram</p>

                    <div className="diagram-container overflow-auto border border-dark border-2 rounded-3">
                        <img src={diagramUrl} alt="generated diagram" className="p-4" />
                    </div>

                    <div className="d-flex justify-content-around mt-4">
                        <Link to="/edit">
                            <button type="button" className="btn btn-success">
                                <i class="fa-solid fa-pen-to-square"></i> Edit
                            </button>
                        </Link>
                        <button type="button" className="btn btn-danger">
                            <i class="fa-regular fa-circle-down"></i> Download
                        </button>
                    </div>
                </div>
            </div>

            <Footer />
        </div>
    );
};

export default Download;
