using AutoMapper;
using MedicalAssistant.Domain.Entities.DoctorsModule;
using MedicalAssistant.Shared.DTOs.DoctorDTOs;

namespace MedicalAssistant.Services.MappingProfiles;

public class DoctorProfile : Profile
{
    public DoctorProfile()
    {
        CreateMap<Doctor, DoctorDTO>()
            .ForMember(dest => dest.Specialty, opt => opt.MapFrom(src => src.Specialty.Name))
            .ForMember(dest => dest.ImageUrl, opt => opt.MapFrom(src => 
                (string.IsNullOrWhiteSpace(src.ImageUrl) || src.ImageUrl == "default-doctor.png") 
                ? "https://cdn-icons-png.flaticon.com/512/3774/3774299.png" 
                : src.ImageUrl))
            .ForMember(dest => dest.YearsExperience, opt => opt.MapFrom(src => src.Experience))
            .ForMember(dest => dest.IsProfileComplete, opt => opt.MapFrom(src =>
                !string.IsNullOrWhiteSpace(src.Bio)
                && !string.IsNullOrWhiteSpace(src.ImageUrl)
                && src.ImageUrl != "default-doctor.png"))
            .ForMember(dest => dest.IsMobileEnabled, opt => opt.MapFrom(src => src.IsAvailable))
            .ForMember(dest => dest.HasSchedule, opt => opt.Ignore());

        CreateMap<Doctor, DoctorDetailsDTO>()
            .IncludeBase<Doctor, DoctorDTO>()  
            .ForMember(dest => dest.Bio, opt => opt.MapFrom(src => src.Bio))
            .ForMember(dest => dest.Experience, opt => opt.MapFrom(src => src.Experience))
            .ForMember(dest => dest.Schedule, opt => opt.Ignore());
    }
}
