import { OrbitControls } from "@react-three/drei";
import { Avatar } from "./Avatar";
import { useControls } from "leva";

export const Experience = () => {
  const { animation } = useControls({
    animation: {
      value: "Idle",
      options: ["Idle", "Angry", "Frustrated", "Stressed", "Happy", "Sad"],
    },
  });

  return (
    <>
      <OrbitControls />
      <group position-y={-1.2} scale={1.2}>
        <Avatar animation={animation} />
      </group>
      <ambientLight intensity={3} />
    </>
  );
};
