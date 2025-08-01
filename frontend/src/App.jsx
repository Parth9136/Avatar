import { Canvas } from "@react-three/fiber";
import { Experience } from "./components/Experience";
import { Leva } from "leva";

function App() {
  return (
    <>
    <Canvas shadows camera={{ position: [0, 2, 5 ], fov: 30 }}>
      <color attach="background" args={["#ececec"]} />
      <Experience />
    </Canvas>
    <Leva hidden /> 
    </>
  );
}

export default App;


