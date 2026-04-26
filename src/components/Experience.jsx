// import React from 'react'

// const Experience = () => {
//   return (
//     <div>Experience</div>
//   )
// }

// export default Experience

import React from "react";
import {
  VerticalTimeline,
  VerticalTimelineElement,
} from "react-vertical-timeline-component";
import { motion } from "framer-motion";

import "react-vertical-timeline-component/style.min.css";

import { styles } from "../styles";
import { experiences } from "../constants";
import { SectionWrapper } from "../hoc";
import { textVariant } from "../utils/motion";

const ExperienceCard = ({ experience }) => {
  return (
    <VerticalTimelineElement
      contentStyle={{
        background: "#1d1836",
        color: "#fff",
      }}
      contentArrowStyle={{ borderRight: "7px solid  #232631" }}
      date={experience.date}
      iconStyle={{ background: experience.iconBg }}
      icon={
        <div className='flex justify-center items-center w-full h-full'>
          <img
            src={experience.icon}
            alt={experience.company_name}
            className='w-[60%] h-[60%] object-contain'
          />
        </div>
      }
    >
      <div>
        <h3 className='text-white text-[24px] font-bold'>{experience.title}</h3>
        <p
          className='text-secondary text-[16px] font-semibold'
          style={{ margin: 0 }}
        >
          {experience.company_name}
        </p>
        {experience.url ? (
          <a
            href={experience.url}
            target='_blank'
            rel='noopener noreferrer'
            className='mt-2 inline-block text-[13px] font-medium text-[#915EFF] hover:text-[#b794ff] transition-colors'
          >
            {experience.linkLabel ?? "Link"}
          </a>
        ) : null}
      </div>

      <div className='mt-5 space-y-3'>
        {experience.points.map((point, index) => (
          <p
            key={`experience-point-${index}`}
            className='text-white-100 text-[14px] leading-relaxed tracking-wide border-l-2 border-[#915EFF]/35 pl-4'
          >
            {point}
          </p>
        ))}
      </div>
    </VerticalTimelineElement>
  );
};

const Experience = () => {
  return (
    <>
      <motion.div variants={textVariant()}>
        <p className={`${styles.sectionSubText} text-center`}>
          Background
        </p>
        <h2 className={`${styles.sectionHeadText} text-center`}>
          Experience & education
        </h2>
      </motion.div>

      <div className='mt-20 flex flex-col'>
        <VerticalTimeline lineColor='#3f3a55'>
          {experiences.map((experience, index) => (
            <ExperienceCard
              key={experience.id ?? `experience-${index}`}
              experience={experience}
            />
          ))}
        </VerticalTimeline>
      </div>
    </>
  );
};

export default SectionWrapper(Experience, "work");