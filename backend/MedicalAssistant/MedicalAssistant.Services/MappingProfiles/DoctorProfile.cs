using AutoMapper;
using AutoMapper;
using MedicalAssistant.Domain.Entities.DoctorsModule;
using MedicalAssistant.Shared.DTOs.DoctorDTOs;

namespace MedicalAssistant.Services.MappingProfiles
{
    public class DoctorProfile : Profile
    {
        public DoctorProfile()
        {
            CreateMap<Doctor, DoctorDTO>()
                .ForMember(dest => dest.FullName, opt => opt.MapFrom(src => src.User.FullName))
                .ForMember(dest => dest.Specialty, opt => opt.MapFrom(src => src.Specialty.Name))
                .ForMember(dest => dest.PhotoUrl, opt => opt.MapFrom(src => src.PhotoUrl ?? string.Empty))
                .ForMember(dest => dest.ConsultFee, opt => opt.MapFrom(src => src.ConsultFee ?? 0));

            CreateMap<Doctor, DoctorDetailsDTO>()
                .IncludeBase<Doctor, DoctorDTO>()
                .ForMember(dest => dest.Bio, opt => opt.MapFrom(src => src.Bio))
                .ForMember(dest => dest.YearsExperience, opt => opt.MapFrom(src => src.YearsExperience))
                .ForMember(dest => dest.License, opt => opt.MapFrom(src => src.License))
                .ForMember(dest => dest.Email, opt => opt.MapFrom(src => src.User.Email));

            CreateMap<DoctorUpdateDto, Doctor>()
                .ForMember(dest => dest.SpecialtyId, opt => opt.MapFrom(src => src.SpecialtyId))
                .ForMember(dest => dest.Bio, opt => opt.MapFrom(src => src.Bio))
                .ForMember(dest => dest.ConsultFee, opt => opt.MapFrom(src => src.ConsultFee))
                .ForMember(dest => dest.YearsExperience, opt => opt.MapFrom(src => src.YearsExperience))
                .ForMember(dest => dest.Id, opt => opt.Ignore())
                .ForMember(dest => dest.UserId, opt => opt.Ignore());
        }
    }
}
