import { motion } from "framer-motion";
import { SectionWrapper } from "../hoc";
import { styles } from "../styles";
import { ComputersCanvas } from "./canvas";
const Hero = () => {
  return (
    <section className={`relative w-full h-96 mx-auto`}>
      <div
        className={`absolute inset-0 top-[120px]  max-w-7xl mx-auto ${styles.paddingX} flex items-start gap-5`}
      >
        <div className='flex flex-col justify-center items-center mt-3'>
          <div className='w-5 h-5 rounded-full bg-[#915EFF]' />
          <div className='w-1 sm:h-80 h-40 violet-gradient' />
        </div>

        <div>
          <h1 className={`${styles.heroHeadText} text-white`}>
            Hi, I'm <span className='text-[#915EFF]'>Kabir Thakur</span>
          </h1>
          <p className={`${styles.heroSubText} mt-2 text-white-100`}>
            I mine gold from data mountains,  <br className='sm:block hidden' />
            crafting nuggets of knowledge that drive strategic decisions
          </p>
        </div>
      </div>


      </section>
  );
};

export default SectionWrapper(Hero,"hero")