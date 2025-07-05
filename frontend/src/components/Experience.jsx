import { OrbitControls } from "@react-three/drei";
import { Avatar } from "./Avatar";
import { useEffect, useState } from "react";

export const Experience = () => {
  const [animation, setAnimation] = useState("Standing");

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch("http://localhost:5000/current_emotion");
        const { emotion } = await res.json();

        const validEmotions = [
          "Amusement", "Awe", "Enthusiasm", "Liking", "Surprised",
          "Angry", "Disgust", "Fear", "Sad", "Standing",
          "Sitting", "Walking", "Running"
        ];

        if (validEmotions.includes(emotion)) {
          setAnimation(emotion);
        } else {
          setAnimation("Standing");
        }
      } catch (err) {
        console.error("Error fetching emotion:", err);
        setAnimation("Standing");
      }
    }, 3000);

    return () => clearInterval(interval);
  }, []);

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





























// import { OrbitControls } from "@react-three/drei";
// import { Avatar } from "./Avatar";
// import { useControls } from "leva";

// export const Experience = () => {
//   const { animation } = useControls({
//     animation: {
//       value: "Standing",
//       options: [
//         "Amusement",
//         "Awe",
//         "Enthusiasm",
//         "Liking",
//         "Surprised",
//         "Angry",
//         "Disgust",
//         "Fear",
//         "Sad",
//         "Standing",
//         "Sitting",
//         "Walking",
//         "Running",
//       ],
//     },
//   });

//   return (
//     <>
//       <OrbitControls />
//       <group position-y={-0.2} scale={1.2}>
//         <Avatar animation={animation} />
//       </group>
//       <ambientLight intensity={3} />
//     </>
//   );
// };
