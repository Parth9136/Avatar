import { OrbitControls } from "@react-three/drei";
import { Avatar } from "./Avatar";
import { useControls } from "leva";

export const Experience = () => {
  const { animation } = useControls({
    animation: {
      value: "Standing",
      options: [
        "Amusement",
        "Awe",
        "Enthusiasm",
        "Liking",
        "Surprised",
        "Angry",
        "Disgust",
        "Fear",
        "Sad",
        "Standing",
        "Sitting",
        "Walking",
        "Running",
      ],
    },
  });

  return (
    <>
      <OrbitControls />
      <group position-y={-0.2} scale={1.2}>
        <Avatar animation={animation} />
      </group>
      <ambientLight intensity={3} />
    </>
  );
};
