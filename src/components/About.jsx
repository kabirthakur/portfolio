import React from 'react';
import { Tilt } from 'react-tilt'
import {motion} from 'framer-motion' ;

import {styles} from '../styles';
import {services} from '../constants';
import { SectionWrapper } from "../hoc";
import { fadeIn, textVariant } from "../utils/motion";

const ServiceCard = ({ index, title, icon }) => (
  <Tilt className='xs:w-[250px] w-full'>
    <motion.div
      variants={fadeIn("right", "spring", index * 0.5, 0.75)}
      className='w-full green-pink-gradient p-[1px] rounded-[20px] shadow-card'
    >
      <div
        options={{
          max: 45,
          scale: 1,
          speed: 450,
        }}
        className='bg-tertiary rounded-[20px] py-5 px-12 min-h-[280px] flex justify-evenly items-center flex-col'
      >
        <img
          src={'.'+icon}
          alt='web-development'
          className='w-40 h-20 object-contain'
        />

        <h3 className='text-white text-[20px] font-bold text-center'>
          {title}
        </h3>
      </div>
    </motion.div>
  </Tilt>
);

const About = () => {
  return (
    <>
      <motion.div variants={textVariant()}>
        
      <h2 className={`${styles.sectionHeadText} text-center`}>About me</h2>

      </motion.div>

      <motion.p
        variants={fadeIn("", "", 0.1, 1)}
        className='mt-4 text-secondary text-[17px] text-center leading-[30px]'
      >
        {/* <p className='text-center mx-auto max-w-xl'> */}
        I'm a proficient Data Scientist and Computational Physicist. I have deep knowledge in Python, R, and SQL, and 
        advanced proficiency in tools like TensorFlow, PyTorch, and PySpark. Leveraging my physics background, 
        I bring a unique analytical perspective to decode complex data patterns. I am adept at swiftly integrating
        new concepts and working collaboratively with clients to design efficient, scalable, and insightful 
        solutions that tackle complex real-world problems. Let's join forces to translate your data into actionable insights!
        {/* </p> */}
      </motion.p>

      <div className='mt-20 flex flex-wrap justify-center gap-10'>
        {services.map((service, index) => (
          <ServiceCard key={service.title} index={index} {...service} />
        ))}
      </div>
    </>
  );
};

export default SectionWrapper(About, "about");