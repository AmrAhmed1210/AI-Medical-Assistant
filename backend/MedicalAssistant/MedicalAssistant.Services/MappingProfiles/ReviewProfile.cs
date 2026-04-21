using AutoMapper;
using MedicalAssistant.Domain.Entities.ReviewsModule;
using MedicalAssistant.Shared.DTOs.ReviewDTOs;

namespace MedicalAssistant.Services.MappingProfiles
{
    public class ReviewMappingProfile : Profile
    {
        public ReviewMappingProfile()
        {
            // Entity -> DTO
            CreateMap<Review, ReviewDto>()
                .ForMember(dest => dest.CreatedAt,
                           opt => opt.MapFrom(src => src.CreatedAt))
                .ForMember(dest => dest.Id,
                           opt => opt.MapFrom(src => src.Id.ToString()));

            // Create DTO -> Entity
            CreateMap<CreateReviewDTO, Review>()
                .ForMember(dest => dest.CreatedAt,
                           opt => opt.MapFrom(src => DateTime.UtcNow))
                .ForMember(dest => dest.Author, 
                           opt => opt.MapFrom(src => src.Author))
                .ForMember(dest => dest.PatientName, 
                           opt => opt.MapFrom(src => src.PatientName));

            // Update DTO -> Entity
            CreateMap<UpdateReviewDto, Review>()
                .ForMember(dest => dest.CreatedAt,
                           opt => opt.Ignore())
                .ForMember(dest => dest.DoctorId,
                           opt => opt.Ignore())
                .ForMember(dest => dest.Author,
                           opt => opt.Ignore());
        }
    }
}
