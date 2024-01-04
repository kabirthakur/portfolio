import { useState } from "react";
import { Tilt } from "react-tilt";
import { motion } from "framer-motion";
import { styles } from "../styles";
import { github } from "../assets";
import { SectionWrapper } from "../hoc";
import { projects } from "../constants";
import { fadeIn, textVariant } from "../utils/motion";

const ProjectCard = ({
  index,
  name,
  category,
  description,
  topics,
  libraries,
  image,
  source_code_link,
  link_labels,
  links,
}) => {
  return (
    <Tilt
      options={{
        max: 20,
        scale: 1,
        speed: 450,
      }}
      className='bg-tertiary p-10 rounded-2xl w-full p-4 sm:p-6 md:p-8 lg:p-10 shadow-card flex sm:flex-row flex-col mb-8'
    >
      <div className='relative w-full sm:w-1/2 h-[300px]'>
        <img
          src={image}
          alt='project_image'
          className='w-full h-full object-fill rounded-2xl'
        />
        <div className='absolute inset-0 flex justify-end m-3 card-img_hover'>
          <div
            onClick={() => window.open(source_code_link, "_blank")}
            className='black-gradient w-10 h-10 rounded-full flex justify-center items-center cursor-pointer'
          >
            <img
              src={github}
              alt='source code'
              className='w-1/2 h-1/2 object-contain'
            />
          </div>
        </div>
      </div>
      <div className='mt-5 sm:mt-0 sm:ml-5 sm:w-1/2'>
        <h3 className='text-white font-bold text-[24px]'>{name}</h3>
        {/* <p className='mt-2 text-secondary text-[14px]'>{category}</p> */}
        <p className='mt-2 text-secondary text-[14px]'>{description}</p>
        <p className='mt-2 text-secondary text-[14px]'>Topics : {topics}</p>
        <p className='mt-2 text-secondary text-[14px]'>{libraries}</p>
        <p className='mt-2'>
          Links :  &nbsp;
          <a href={links.l1} target="_blank" rel="noreferrer">
            {link_labels.label1}
          </a>
          {links.l2 && // If links.l2 is available, render the comma and the link
            <>
              ,&nbsp;
              <a href={links.l2} target="_blank" rel="noreferrer">
                {link_labels.label2}
              </a>
            </>
          }
        </p>
      </div>
    </Tilt>
  );
};

const Works = () => {
  const [selectedCategory, setSelectedCategory] = useState('All');
  const allCategories = projects.flatMap(project => project.category);
  const uniqueCategories = ['All', ...new Set(allCategories)];

  const filteredProjects = selectedCategory === 'All'
    ? projects
    : projects.filter(project => project.category.includes(selectedCategory));


  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-center">
      <p className={`${styles.sectionSubText}`}>My work</p>
      <h2 className={`${styles.sectionHeadText}`}>Projects.</h2>

      <div className='w-full flex flex-col items-center'>
        <p className='mt-3 text-secondary text-[17px] max-w-3xl leading-[30px]'>
          Following projects showcase my skills and experience through
          real-world examples of my work. Each project is briefly described with
          links to code repositories and live demos in it. It reflects my
          ability to solve complex problems, work with different technologies,
          and manage projects effectively.
        </p>
      </div>
    <br></br>
      <div className="bg-tertiary rounded-2xl p-4 flex justify-center space-x-4">
        {uniqueCategories.map(category => (
          <button
            key={category}
            className={`text-white ${selectedCategory === category ? 'font-bold' : ''}`}
            onClick={() => setSelectedCategory(category)}
          >
            {category}
          </button>
        ))}
      </div>

      <div className='mt-20 flex flex-wrap justify-center'>
        {filteredProjects.map((project, index) => (
          <ProjectCard key={`project-${index}`} index={index} {...project} />
        ))}
      </div>
    </div>
  );
};

export default SectionWrapper(Works, "projects");
